using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Emit;
using Microsoft.CSharp.RuntimeBinder;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

public class TestCase
{
    public string method_name { get; set; }
    public string CSharpArg { get; set; }
    public object Output { get; set; }
}

public class DataItem
{
    public string apl { get; set; }
    public string csharp { get; set; }
    public string model_pred { get; set; }

    [JsonConverter(typeof(SingleOrArrayConverter<TestCase>))]
    public List<TestCase> io { get; set; }
}
    public class TestResult
{
    public int TotalTests { get; set; }
    public int PassedTests { get; set; }
    public int FailedTests { get; set; }
    public List<string> FailureMessages { get; set; } = new List<string>();
    public bool AllPassed => FailedTests == 0;
    public bool CompiledSuccessfully { get; set; }
}

public class AggregateTestStats
{
    public int TotalSubmissions { get; set; }
    public int CompiledSubmissions { get; set; }
    public int PartialPassSubmissions { get; set; } // At least 1 test passed
    public int FullPassSubmissions { get; set; }    // All tests passed
    
    public double CompileRate => TotalSubmissions > 0 
        ? (double)CompiledSubmissions / TotalSubmissions * 100 
        : 0;
    
    public double PartialPassRate => CompiledSubmissions > 0 
        ? (double)PartialPassSubmissions / CompiledSubmissions * 100 
        : 0;
    
    public double FullPassRate => CompiledSubmissions > 0 
        ? (double)FullPassSubmissions / CompiledSubmissions * 100 
        : 0;
}

public class DynamicCodeTester
{
    private static bool IsValueTupleType(Type t)
{
    return t.IsValueType &&
           t.IsGenericType &&
           t.FullName.StartsWith("System.ValueTuple`", StringComparison.Ordinal);
}
private static string FormatValueTuple(object tuple)
{
    if (tuple == null) return "null";

    var type = tuple.GetType();
    var fields = type.GetFields(); // Item1, Item2, ...
    var parts = new List<string>();

    foreach (var f in fields)
    {
        var val = f.GetValue(tuple);
        parts.Add(FormatOutput(val));   // recursive
    }

    return "(" + string.Join(", ", parts) + ")";
}
private static object ConvertJsonArrayToValueTuple(JsonElement element, Type tupleType)
{
    if (element.ValueKind != JsonValueKind.Array)
        throw new InvalidOperationException("Expected JSON array to convert to ValueTuple.");

    var genericArgs = tupleType.GetGenericArguments();
    var items = element.EnumerateArray().ToList();

    if (items.Count != genericArgs.Length)
        throw new InvalidOperationException(
            $"JSON array length {items.Count} does not match tuple arity {genericArgs.Length}.");

    object[] values = new object[genericArgs.Length];
    for (int i = 0; i < genericArgs.Length; i++)
    {
        values[i] = ConvertJsonElement(items[i], genericArgs[i]);
    }

    return Activator.CreateInstance(tupleType, values);
}

private static MethodInfo CloseGenericMethodIfNeeded(MethodInfo method, Type[] argTypes)
{
    if (!method.IsGenericMethodDefinition)
        return method;

    var genericParams = method.GetGenericArguments();
    var map = new Dictionary<Type, Type>();

    var paramTypes = method.GetParameters().Select(p => p.ParameterType).ToArray();
    for (int i = 0; i < paramTypes.Length && i < argTypes.Length; i++)
    {
        InferGenericArgumentsFrom(paramTypes[i], argTypes[i], map);
    }

    var typeArgs = genericParams
        .Select(gp => map.TryGetValue(gp, out var t) ? t : typeof(object))
        .ToArray();

    return method.MakeGenericMethod(typeArgs);
}

private static void InferGenericArgumentsFrom(Type paramType, Type argType, Dictionary<Type, Type> map)
{
    if (paramType.IsGenericParameter)
    {
        if (!map.ContainsKey(paramType))
            map[paramType] = argType;
        return;
    }

    // Arrays: T[] vs int[]
    if (paramType.IsArray && argType.IsArray)
    {
        InferGenericArgumentsFrom(paramType.GetElementType(), argType.GetElementType(), map);
        return;
    }

    if (paramType.IsGenericType && argType.IsGenericType &&
        paramType.GetGenericTypeDefinition() == argType.GetGenericTypeDefinition())
    {
        var pArgs = paramType.GetGenericArguments();
        var aArgs = argType.GetGenericArguments();
        for (int i = 0; i < pArgs.Length && i < aArgs.Length; i++)
        {
            InferGenericArgumentsFrom(pArgs[i], aArgs[i], map);
        }
    }
}
private static string WrapCodeIfNeeded(string code)
{
    code = code.Trim();
    
    var tree = CSharpSyntaxTree.ParseText(code);
    var root = tree.GetRoot();
    
    // Check if there's already a type declaration (class, struct, interface, etc.)
    var hasTypeDeclaration = root.DescendantNodes()
        .Any(node => node is Microsoft.CodeAnalysis.CSharp.Syntax.TypeDeclarationSyntax);
    
    if (hasTypeDeclaration)
    {
        return code;
    }
    
    // No type declaration found - wrap it
    return $@"
public class GeneratedSolution
{{
    {code}
}}";
}

public static Assembly CompileCode(string code)
{
    string header = @"
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using System.Threading.Tasks;    
using System.Threading;   
";
    string processedCode = WrapCodeIfNeeded(code);

    var syntaxTree = CSharpSyntaxTree.ParseText(header + Environment.NewLine + processedCode);

    var runtimePath = Path.GetDirectoryName(typeof(object).Assembly.Location);

    var references = new List<MetadataReference>
    {
        MetadataReference.CreateFromFile(typeof(object).Assembly.Location),
        MetadataReference.CreateFromFile(typeof(Console).Assembly.Location),
        MetadataReference.CreateFromFile(typeof(Enumerable).Assembly.Location),
        MetadataReference.CreateFromFile(Assembly.Load("System.Runtime").Location),
        MetadataReference.CreateFromFile(Assembly.Load("System.Collections").Location),
        MetadataReference.CreateFromFile(Assembly.Load("System.Linq").Location),
        MetadataReference.CreateFromFile(typeof(System.Threading.Tasks.Parallel).Assembly.Location),
        MetadataReference.CreateFromFile(typeof(System.Threading.Thread).Assembly.Location),
        MetadataReference.CreateFromFile(typeof(System.Text.RegularExpressions.Regex).Assembly.Location),
        MetadataReference.CreateFromFile(typeof(System.Text.Json.JsonSerializer).Assembly.Location),
        
        MetadataReference.CreateFromFile(Path.Combine(runtimePath, "System.Linq.Expressions.dll")),
        MetadataReference.CreateFromFile(Path.Combine(runtimePath, "System.Runtime.Extensions.dll")),
        MetadataReference.CreateFromFile(Path.Combine(runtimePath, "System.ObjectModel.dll")),
        MetadataReference.CreateFromFile(Path.Combine(runtimePath, "netstandard.dll")),
    };

    // Add System.Numerics references
    try
    {
        var numericsAssembly = Assembly.Load("System.Numerics");
        references.Add(MetadataReference.CreateFromFile(numericsAssembly.Location));
    }
    catch
    {
        try
        {
            // Fallback: try loading by type
            references.Add(MetadataReference.CreateFromFile(typeof(System.Numerics.Complex).Assembly.Location));
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Could not load System.Numerics.Complex: {ex.Message}");
        }
        
        try
        {
            references.Add(MetadataReference.CreateFromFile(typeof(System.Numerics.BigInteger).Assembly.Location));
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Could not load System.Numerics.BigInteger: {ex.Message}");
        }
    }

    // Add Microsoft.CSharp
    try
    {
        var msCSharpPath = Path.Combine(runtimePath, "Microsoft.CSharp.dll");
        if (File.Exists(msCSharpPath))
        {
            references.Add(MetadataReference.CreateFromFile(msCSharpPath));
        }
        else
        {
            var loadedAssembly = AppDomain.CurrentDomain.GetAssemblies()
                .FirstOrDefault(a => a.GetName().Name == "Microsoft.CSharp");
            if (loadedAssembly != null)
            {
                references.Add(MetadataReference.CreateFromFile(loadedAssembly.Location));
            }
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Warning: Could not load Microsoft.CSharp: {ex.Message}");
    }

    var compilation = CSharpCompilation.Create(
        $"DynamicAssembly_{Guid.NewGuid()}",
        new[] { syntaxTree },
        references,
        new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary)
    );

    using var ms = new MemoryStream();
    EmitResult result = compilation.Emit(ms);

    if (!result.Success)
    {
        var errors = string.Join("\n", result.Diagnostics
            .Where(d => d.Severity == DiagnosticSeverity.Error)
            .Select(d => $"  - {d.GetMessage()}"));
        throw new Exception($"Compilation failed:\n{errors}");
    }

    ms.Seek(0, SeekOrigin.Begin);
    return Assembly.Load(ms.ToArray());
}
public static (Assembly assembly, byte[] bytes) CompileCodeWithBytes(string code)
{
    string header = @"
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using System.Threading.Tasks;    
using System.Threading;   
";
    string processedCode = WrapCodeIfNeeded(code);

    var syntaxTree = CSharpSyntaxTree.ParseText(header + Environment.NewLine + processedCode);

    var runtimePath = Path.GetDirectoryName(typeof(object).Assembly.Location);

    var references = new List<MetadataReference>
    {
        MetadataReference.CreateFromFile(typeof(object).Assembly.Location),
        MetadataReference.CreateFromFile(typeof(Console).Assembly.Location),
        MetadataReference.CreateFromFile(typeof(Enumerable).Assembly.Location),
        MetadataReference.CreateFromFile(Assembly.Load("System.Runtime").Location),
        MetadataReference.CreateFromFile(Assembly.Load("System.Collections").Location),
        MetadataReference.CreateFromFile(Assembly.Load("System.Linq").Location),
        MetadataReference.CreateFromFile(typeof(System.Threading.Tasks.Parallel).Assembly.Location),
        MetadataReference.CreateFromFile(typeof(System.Threading.Thread).Assembly.Location),
        MetadataReference.CreateFromFile(typeof(System.Text.RegularExpressions.Regex).Assembly.Location),
        MetadataReference.CreateFromFile(typeof(System.Numerics.Complex).Assembly.Location),
        MetadataReference.CreateFromFile(typeof(System.Numerics.BigInteger).Assembly.Location),
        MetadataReference.CreateFromFile(typeof(System.Text.Json.JsonSerializer).Assembly.Location),
        
        MetadataReference.CreateFromFile(Path.Combine(runtimePath, "System.Linq.Expressions.dll")),
        MetadataReference.CreateFromFile(Path.Combine(runtimePath, "System.Runtime.Extensions.dll")),
        MetadataReference.CreateFromFile(Path.Combine(runtimePath, "System.ObjectModel.dll")),
        MetadataReference.CreateFromFile(Path.Combine(runtimePath, "netstandard.dll")),
    };

    try
    {
        var msCSharpPath = Path.Combine(runtimePath, "Microsoft.CSharp.dll");
        if (File.Exists(msCSharpPath))
        {
            references.Add(MetadataReference.CreateFromFile(msCSharpPath));
        }
        else
        {
            var loadedAssembly = AppDomain.CurrentDomain.GetAssemblies()
                .FirstOrDefault(a => a.GetName().Name == "Microsoft.CSharp");
            if (loadedAssembly != null)
            {
                references.Add(MetadataReference.CreateFromFile(loadedAssembly.Location));
            }
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Warning: Could not load Microsoft.CSharp: {ex.Message}");
    }

    var compilation = CSharpCompilation.Create(
        $"DynamicAssembly_{Guid.NewGuid()}",
        new[] { syntaxTree },
        references,
        new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary)
    );

    using var ms = new MemoryStream();
    EmitResult result = compilation.Emit(ms);

    if (!result.Success)
    {
        var errors = string.Join("\n", result.Diagnostics
            .Where(d => d.Severity == DiagnosticSeverity.Error)
            .Select(d => $"  - {d.GetMessage()}"));
        throw new Exception($"Compilation error:\n{errors}");
    }

    ms.Seek(0, SeekOrigin.Begin);
    byte[] assemblyBytes = ms.ToArray();
    return (Assembly.Load(assemblyBytes), assemblyBytes);
}

public static object[] EvaluateCSharpArguments(string expression, Assembly codeAssembly = null, byte[] codeAssemblyBytes = null)
{

    if (codeAssembly == null && codeAssemblyBytes != null)
    {
        codeAssembly = Assembly.Load(codeAssemblyBytes);
    }

    string wrapperCode = $@"
        using System;
        using System.Linq;
        using System.Collections.Generic;

        public class ExpressionEvaluator
        {{
            public static object[] Evaluate()
            {{
                return new object[] {{ {expression} }};
            }}
        }}
    ";

    string header = @"
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
";
    string processedCode = WrapCodeIfNeeded(wrapperCode);
    var syntaxTree = CSharpSyntaxTree.ParseText(header + Environment.NewLine + processedCode);

    var runtimePath = Path.GetDirectoryName(typeof(object).Assembly.Location);

    var references = new List<MetadataReference>
    {
        MetadataReference.CreateFromFile(typeof(object).Assembly.Location),
        MetadataReference.CreateFromFile(typeof(Console).Assembly.Location),
        MetadataReference.CreateFromFile(typeof(Enumerable).Assembly.Location),
        MetadataReference.CreateFromFile(Assembly.Load("System.Runtime").Location),
        MetadataReference.CreateFromFile(Assembly.Load("System.Collections").Location),
        MetadataReference.CreateFromFile(Assembly.Load("System.Linq").Location),
    };

    // Add reference to the code assembly if provided
    if (codeAssemblyBytes != null)
    {
        references.Add(MetadataReference.CreateFromImage(codeAssemblyBytes));
    }

    var compilation = CSharpCompilation.Create(
        $"EvalAssembly_{Guid.NewGuid()}",
        new[] { syntaxTree },
        references,
        new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary)
    );

    using var ms = new MemoryStream();
    EmitResult result = compilation.Emit(ms);

    if (!result.Success)
    {
        var errors = string.Join("\n", result.Diagnostics
            .Where(d => d.Severity == DiagnosticSeverity.Error)
            .Select(d => $"  - {d.GetMessage()}"));
        throw new Exception($"Compilation error:\n{errors}");
    }

    ms.Seek(0, SeekOrigin.Begin);
    var evalAssembly = Assembly.Load(ms.ToArray());
    
    var type = evalAssembly.GetTypes().First(t => t.Name == "ExpressionEvaluator");
    var method = type.GetMethod("Evaluate");
    return (object[])method.Invoke(null, null);
}
public static TestResult TestCode(string code, List<TestCase> testCases, int codeIndex)
{
    var result = new TestResult { TotalTests = testCases.Count, CompiledSuccessfully = false };

    try
    {
        // Compile and get both assembly and bytes
        var (assembly, assemblyBytes) = CompileCodeWithBytes(code);
        result.CompiledSuccessfully = true;

        var type = assembly.GetTypes()
            .Where(t => t.IsClass
                        && !t.Name.Contains("<")
                        && (!t.IsAbstract || t.IsSealed))
            .FirstOrDefault();

        if (type == null)
        {
            result.FailedTests = testCases.Count;
            result.FailureMessages.Add("No public class found in code");
            return result;
        }

        // Set up assembly resolver for this test run
        ResolveEventHandler resolver = (sender, args) =>
        {
            // Check if the requested assembly is our dynamically compiled code
            var requestedName = new AssemblyName(args.Name);
            var ourName = assembly.GetName();
            
            if (requestedName.Name == ourName.Name)
            {
                return assembly;
            }
            return null;
        };

        // Register the resolver
        AppDomain.CurrentDomain.AssemblyResolve += resolver;

        try
        {
            for (int i = 0; i < testCases.Count; i++)
            {
                var testCase = testCases[i];
                try
                {
                    // Pass both assembly and bytes to evaluation
                    object[] args = EvaluateCSharpArguments(testCase.CSharpArg, assembly, assemblyBytes);
                    Type[] argTypes = args
                        .Select(a => a?.GetType() ?? typeof(object))
                        .ToArray();

                    MethodInfo method = FindMethod(type, testCase.method_name, argTypes);

                    if (method == null)
                    {
                        Console.WriteLine("Methods on type " + type.FullName + ":");
                        foreach (var m in type.GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance))
                        {
                            Console.WriteLine("  " + m);
                        }
                        Console.WriteLine("method_name: " + (testCase.method_name ?? "null"));
                        Console.WriteLine("argTypes: " + string.Join(", ", argTypes.Select(t => t?.ToString() ?? "null")));

                        result.FailureMessages.Add($"  Test {i + 1}: No suitable method found");
                        result.FailedTests++;
                        continue;
                    }
                    method = CloseGenericMethodIfNeeded(method, argTypes);

                    object instance = method.IsStatic ? null : Activator.CreateInstance(type);
                    object methodResult = method.Invoke(instance, args);

                    var expectedOutput = testCase.Output;

                    if (expectedOutput is JsonElement jsonElement)
                    {
                        expectedOutput = ConvertJsonElement(jsonElement, methodResult?.GetType());
                    }

                    if (CompareObjects(methodResult, expectedOutput))
                    {
                        result.PassedTests++;
                    }
                    else
                    {
                        result.FailureMessages.Add(
                            $"  Test {i + 1}: Functional error, expected {FormatOutput(expectedOutput)}, got {FormatOutput(methodResult)}");
                        result.FailedTests++;
                    }
                }
                catch (Exception ex)
                {
                    result.FailureMessages.Add($"  Test {i + 1}: Run time error, {ex.InnerException?.Message ?? ex.Message}");
                    result.FailedTests++;
                }
            }
        }
        finally
        {
            // Clean up the resolver
            AppDomain.CurrentDomain.AssemblyResolve -= resolver;
        }
    }
    catch (Exception ex)
    {
        result.FailedTests = testCases.Count;
        result.FailureMessages.Add($"Compilation error: {ex.Message}");
        result.CompiledSuccessfully = false;
    }

    return result;
}

private static MethodInfo FindMethod(Type type, string methodName, Type[] argTypes)
{
    var allMethods = type.GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance)
        .Where(m => !m.IsSpecialName && m.DeclaringType == type)
        .ToList();

    if (allMethods.Count == 0)
        return null;

    // Start with all methods as candidates
    var methods = allMethods;

    // If we have a method name, try to filter by it
    if (!string.IsNullOrEmpty(methodName))
    {
        var byName = methods.Where(m => m.Name == methodName).ToList();

        if (byName.Count > 0)
        {
            methods = byName;
        }
    }

    // If after optional filtering we have exactly one candidate
    if (methods.Count == 1 && (argTypes == null || argTypes.Length == methods[0].GetParameters().Length))
    {
        return methods[0];
    }

    // If we have argument types, try to match on parameter count + compatibility
    if (argTypes != null && argTypes.Length > 0)
    {
        MethodInfo best = null;

        foreach (var m in methods)
        {
            var ps = m.GetParameters();
            if (ps.Length != argTypes.Length)
                continue;

            bool compatible = true;
            bool allExact = true;

            for (int i = 0; i < ps.Length; i++)
            {
                var paramType = ps[i].ParameterType;
                var argType = argTypes[i] ?? typeof(object);

                // Special-case generic array parameters like T[]:
                if (paramType.IsArray &&
                    paramType.GetElementType() != null &&
                    paramType.GetElementType().IsGenericParameter &&
                    argType.IsArray &&
                    argType.GetArrayRank() == paramType.GetArrayRank())
                {
                    allExact = false; // compatible but not exact
                    continue;
                }

                if (!paramType.IsAssignableFrom(argType))
                {
                    compatible = false;
                    break;
                }

                if (paramType != argType)
                    allExact = false;
            }

            if (!compatible)
                continue;

            if (allExact)
                return m;    // perfect match

            if (best == null)
                best = m;    // remember first compatible as fallback
        }

        if (best != null)
            return best;
    }

    // Fallback: if no argTypes or no good match, just pick the first candidate
    return methods.FirstOrDefault();
}

    // Backwards-compatible wrapper if you ever call it with a single type
    private static MethodInfo FindMethod(Type type, string methodName, Type inputType)
    {
        return FindMethod(type, methodName, new[] { inputType });
    }

    private static object ConvertJsonElement(JsonElement element, Type targetType)
{
    if (targetType == null) targetType = typeof(object);

    switch (element.ValueKind)
    {
        case JsonValueKind.Number:
            if (targetType == typeof(int)) return element.GetInt32();
            if (targetType == typeof(long)) return element.GetInt64();
            if (targetType == typeof(double)) return element.GetDouble();
            if (targetType == typeof(float)) return element.GetSingle();
            return element.GetDecimal();

        case JsonValueKind.String:
            return element.GetString();

        case JsonValueKind.True:
        case JsonValueKind.False:
            return element.GetBoolean();

        case JsonValueKind.Array:
            if (!targetType.IsArray &&
            targetType != typeof(object) &&
            element.GetArrayLength() == 1)
            {
                var inner = element.EnumerateArray().First();
                return ConvertJsonElement(inner, targetType);
            }
            if (targetType.IsArray)
            {
                var elemType = targetType.GetElementType();
                int rank = targetType.GetArrayRank();

                // 1D array: [a, b, c]
                if (rank == 1)
                {
                    var items = element.EnumerateArray().ToList();
                    var arr = Array.CreateInstance(elemType, items.Count);
                    for (int i = 0; i < items.Count; i++)
                    {
                        var v = ConvertJsonElement(items[i], elemType);
                        arr.SetValue(v, i);
                    }
                    return arr;
                }

                // 2D array: [[...], [...], ...]
                if (rank == 2)
                {
                    var rows = element.EnumerateArray().ToList();
                    int rowCount = rows.Count;
                    int colCount = rowCount > 0 ? rows[0].GetArrayLength() : 0;

                    var arr2d = Array.CreateInstance(elemType, rowCount, colCount);

                    for (int r = 0; r < rowCount; r++)
                    {
                        var row = rows[r];
                        int c = 0;
                        foreach (var cell in row.EnumerateArray())
                        {
                            var v = ConvertJsonElement(cell, elemType);
                            arr2d.SetValue(v, r, c);
                            c++;
                        }
                    }
                    return arr2d;
                }
            }
            // Handle List<T> types
            if (targetType.IsGenericType && targetType.GetGenericTypeDefinition() == typeof(List<>))
            {
                var elemType = targetType.GetGenericArguments()[0];
                var items = element.EnumerateArray().ToList();
                
                var listType = typeof(List<>).MakeGenericType(elemType);
                var list = (System.Collections.IList)Activator.CreateInstance(listType);
                
                foreach (var item in items)
                {
                    var v = ConvertJsonElement(item, elemType);
                    list.Add(v);
                }
                return list;
            }
            // Handle a single tuple target like (int,int)
            if (IsValueTupleType(targetType))
            {
                return ConvertJsonArrayToValueTuple(element, targetType);
            }
            if (targetType == typeof(object))
            {
                var items = element.EnumerateArray().ToList();
                var arr = new object[items.Count];
                for (int i = 0; i < items.Count; i++)
                {
                    arr[i] = ConvertJsonElement(items[i], typeof(object));
                }
                return arr;
            }

            // Fallback: return object[] (changed from List<object>)
            {
                var items = element.EnumerateArray().ToList();
                var arr = new object[items.Count];
                for (int i = 0; i < items.Count; i++)
                {
                    arr[i] = ConvertJsonElement(items[i], typeof(object));
                }
                return arr;
            }

        default:
            // Objects or anything else – last resort
             return element.ToString(); 
    }
}
private static bool CompareMultiDimAndJagged(Array multiDim, Array jagged, int[] dimLengths = null)
{
    // Build dimension info from the multi-dimensional array
    if (dimLengths == null)
    {
        dimLengths = new int[multiDim.Rank];
        for (int i = 0; i < multiDim.Rank; i++)
        {
            dimLengths[i] = multiDim.GetLength(i);
        }
    }

    // Base case: if we're at the innermost dimension, compare directly
    if (multiDim.Rank == 1)
    {
        if (jagged.Rank != 1 || jagged.Length != multiDim.GetLength(0))
            return false;

        for (int i = 0; i < jagged.Length; i++)
        {
            if (!CompareObjects(multiDim.GetValue(i), jagged.GetValue(i)))
                return false;
        }
        return true;
    }

    // Recursive case: jagged should be 1D array where each element is an array
    if (jagged.Rank != 1 || jagged.Length != dimLengths[0])
        return false;

    // Compare each "slice"
    for (int i = 0; i < dimLengths[0]; i++)
    {
        var jaggedElement = jagged.GetValue(i);
        
        // Extract the slice from the multi-dimensional array
        if (multiDim.Rank == 2)
        {
            // For 2D, extract a row
            if (!(jaggedElement is Array jaggedRow) || jaggedRow.Length != dimLengths[1])
                return false;

            for (int j = 0; j < dimLengths[1]; j++)
            {
                if (!CompareObjects(multiDim.GetValue(i, j), jaggedRow.GetValue(j)))
                    return false;
            }
        }
        else if (multiDim.Rank == 3)
        {
            // For 3D, extract a 2D slice
            if (!(jaggedElement is Array jaggedSlice) || jaggedSlice.Length != dimLengths[1])
                return false;

            for (int j = 0; j < dimLengths[1]; j++)
            {
                var jaggedRow = jaggedSlice.GetValue(j);
                if (!(jaggedRow is Array jaggedRowArray) || jaggedRowArray.Length != dimLengths[2])
                    return false;

                for (int k = 0; k < dimLengths[2]; k++)
                {
                    if (!CompareObjects(multiDim.GetValue(i, j, k), jaggedRowArray.GetValue(k)))
                        return false;
                }
            }
        }
        else
        {
            // For higher dimensions, we'd need a more general recursive approach
            // For now, fall back to false
            return false;
        }
    }

    return true;
}
private static bool CompareObjects(object a, object b)
{
    if (a == null && b == null) return true;
    if (a == null || b == null) return false;

    var typeA = a.GetType();
    var typeB = b.GetType();

    // 1) Deep compare ValueTuples
    if (IsValueTupleType(typeA) && IsValueTupleType(typeB))
    {
        var fieldsA = typeA.GetFields();
        var fieldsB = typeB.GetFields();

        if (fieldsA.Length != fieldsB.Length) return false;

        for (int i = 0; i < fieldsA.Length; i++)
        {
            var vA = fieldsA[i].GetValue(a);
            var vB = fieldsB[i].GetValue(b);
            if (!CompareObjects(vA, vB))
                return false;
        }
        return true;
    }

    // 1b) Deep compare System.Tuple (Tuple<T1>, Tuple<T1,T2>, etc.)
    if (typeA.IsGenericType && typeA.FullName.StartsWith("System.Tuple`", StringComparison.Ordinal) &&
        typeB.IsGenericType && typeB.FullName.StartsWith("System.Tuple`", StringComparison.Ordinal))
    {
        var propsA = typeA.GetProperties()
            .Where(p => p.Name.StartsWith("Item"))
            .OrderBy(p => p.Name)
            .ToList();
        var propsB = typeB.GetProperties()
            .Where(p => p.Name.StartsWith("Item"))
            .OrderBy(p => p.Name)
            .ToList();

        if (propsA.Count != propsB.Count) return false;

        for (int i = 0; i < propsA.Count; i++)
        {
            var vA = propsA[i].GetValue(a);
            var vB = propsB[i].GetValue(b);
            if (!CompareObjects(vA, vB))
                return false;
        }
        return true;
    }

    // Handle tuple vs array comparison (e.g., Tuple<int[], int[]> vs jagged array [[3,4], [2]])
    if (typeA.IsGenericType && typeA.FullName.StartsWith("System.Tuple`", StringComparison.Ordinal) && b is Array arrB)
    {
        var propsA = typeA.GetProperties()
            .Where(p => p.Name.StartsWith("Item"))
            .OrderBy(p => p.Name)
            .ToList();

        if (arrB.Rank != 1 || arrB.Length != propsA.Count)
            return false;

        for (int i = 0; i < propsA.Count; i++)
        {
            var vA = propsA[i].GetValue(a);
            var vB = arrB.GetValue(i);
            if (!CompareObjects(vA, vB))
                return false;
        }
        return true;
    }

    // Handle array vs tuple comparison (symmetric case)
    if (a is Array arrA && typeB.IsGenericType && typeB.FullName.StartsWith("System.Tuple`", StringComparison.Ordinal))
    {
        return CompareObjects(b, a); // Swap and use the logic above
    }

    // 2) Arrays (1D and multi-D)
    if (a is Array arrA2 && b is Array arrB2)
    {
        // Handle multi-dimensional vs jagged equivalence
        if (arrA2.Rank != arrB2.Rank)
        {
            // Try comparing multi-dim with jagged
            if (arrA2.Rank > 1 && arrB2.Rank == 1)
            {
                if (CompareMultiDimAndJagged(arrA2, arrB2))
                    return true;
            }
            if (arrB2.Rank > 1 && arrA2.Rank == 1)
            {
                if (CompareMultiDimAndJagged(arrB2, arrA2))
                    return true;
            }

            return false;
        }

        // Same rank → check dimensions and compare elements
        for (int dim = 0; dim < arrA2.Rank; dim++)
        {
            if (arrA2.GetLength(dim) != arrB2.GetLength(dim))
                return false;
        }

        int[] indices = new int[arrA2.Rank];
        return CompareArrayElements(arrA2, arrB2, indices, 0);
    }

    // 3) Handle IEnumerable (List, etc.) - compare element by element
    if (a is System.Collections.IEnumerable enumA && 
        b is System.Collections.IEnumerable enumB &&
        !(a is string) && !(b is string) && // strings are IEnumerable but should use Equals
        !(a is Array) && !(b is Array)) // arrays handled above
    {
        var enumeratorA = enumA.GetEnumerator();
        var enumeratorB = enumB.GetEnumerator();

        while (true)
        {
            bool hasNextA = enumeratorA.MoveNext();
            bool hasNextB = enumeratorB.MoveNext();

            // Different lengths
            if (hasNextA != hasNextB)
                return false;

            // Both ended at same time
            if (!hasNextA)
                return true;

            // Compare current elements
            if (!CompareObjects(enumeratorA.Current, enumeratorB.Current))
                return false;
        }
    }

    // 4) Numeric tolerance
    if (IsNumeric(a) && IsNumeric(b))
    {
        double valA = Convert.ToDouble(a);
        double valB = Convert.ToDouble(b);
        return Math.Abs(valA - valB) < 0.0001;
    }

    // 5) Fallback: normal Equals
    return a.Equals(b);
}


    private static bool CompareArrayElements(Array arrA, Array arrB, int[] indices, int dimension)
    {
        if (dimension == indices.Length)
        {
            var vA = arrA.GetValue(indices);
            var vB = arrB.GetValue(indices);
            return CompareObjects(vA, vB);
        }

        int len = arrA.GetLength(dimension);
        for (int i = 0; i < len; i++)
        {
            indices[dimension] = i;
            if (!CompareArrayElements(arrA, arrB, indices, dimension + 1))
                return false;
        }

        return true;
    }

    private static bool IsNumeric(object obj)
    {
        return obj is int || obj is long || obj is float || obj is double || obj is decimal;
    }

private static string FormatOutput(object obj)
{
    if (obj == null) return "null";

    var t = obj.GetType();

    // NEW: ValueTuple formatting
    if (IsValueTupleType(t))
    {
        return FormatValueTuple(obj);
    }

    // Arrays (including arrays of tuples)
    if (obj is Array arr)
    {
        // Multi-dimensional
        if (arr.Rank > 1)
        {
            var rows = new List<string>();
            int rowsCount = arr.GetLength(0);
            int colsCount = arr.GetLength(1);

            for (int i = 0; i < rowsCount; i++)
            {
                var rowItems = new List<string>();
                for (int j = 0; j < colsCount; j++)
                {
                    rowItems.Add(FormatOutput(arr.GetValue(i, j)));
                }
                rows.Add("[" + string.Join(", ", rowItems) + "]");
            }

            return "[" + string.Join(", ", rows) + "]";
        }

        // 1D array
        var items = new List<string>();
        foreach (var item in arr)
        {
            items.Add(FormatOutput(item));
        }
        return "[" + string.Join(", ", items) + "]";
    }

    return obj.ToString();
}

}


public class SingleOrArrayConverter<T> : JsonConverter<List<T>>
{
    public override List<T> Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        var result = new List<T>();

        if (reader.TokenType == JsonTokenType.StartArray)
        {
            // Normal array case
            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndArray)
                    break;

                var item = JsonSerializer.Deserialize<T>(ref reader, options);
                result.Add(item);
            }
        }
        else if (reader.TokenType == JsonTokenType.StartObject)
        {
            // Single object -> wrap into list
            var item = JsonSerializer.Deserialize<T>(ref reader, options);
            result.Add(item);
        }
        else if (reader.TokenType == JsonTokenType.Null)
        {
            // null => empty list
        }
        else
        {
            throw new JsonException($"Unexpected token {reader.TokenType} when parsing List<{typeof(T).Name}>");
        }

        return result;
    }

    public override void Write(Utf8JsonWriter writer, List<T> value, JsonSerializerOptions options)
    {
        writer.WriteStartArray();
        foreach (var item in value)
        {
            JsonSerializer.Serialize(writer, item, options);
        }
        writer.WriteEndArray();
    }
}

public class ExecutorInput
{
    public string code { get; set; }
    public List<TestCase> testCases { get; set; }
}

public class ExecutorOutput
{
    public int TotalTests { get; set; }
    public int PassedTests { get; set; }
    public int FailedTests { get; set; }
    public List<string> FailureMessages { get; set; } = new List<string>();
    public bool AllPassed => FailedTests == 0;
    public bool CompiledSuccessfully { get; set; }
}

// Note: TestCase and other classes are already defined

class Program
{
    static void Main(string[] args)
    {
        try
        {
            // Read JSON input from stdin
            string input = Console.In.ReadToEnd();
            
            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };
            
            var executorInput = JsonSerializer.Deserialize<ExecutorInput>(input, options);
            
            if (executorInput == null)
            {
                Console.Error.WriteLine("Failed to parse input JSON");
                Environment.Exit(1);
                return;
            }
            
            // Use the original DynamicCodeTester.TestCode method
            var testResult = DynamicCodeTester.TestCode(
                executorInput.code, 
                executorInput.testCases, 
                0  // codeIndex
            );
            
            // Convert TestResult to ExecutorOutput
            var output = new ExecutorOutput
            {
                TotalTests = testResult.TotalTests,
                PassedTests = testResult.PassedTests,
                FailedTests = testResult.FailedTests,
                FailureMessages = testResult.FailureMessages,
                CompiledSuccessfully = testResult.CompiledSuccessfully
            };
            
            // Output result as JSON to stdout
            string outputJson = JsonSerializer.Serialize(output, new JsonSerializerOptions
            {
                WriteIndented = false
            });
            
            Console.WriteLine(outputJson);
        }
        catch (Exception ex)
        {
            var errorResult = new ExecutorOutput
            {
                TotalTests = 0,
                PassedTests = 0,
                FailedTests = 0,
                FailureMessages = new List<string> { $"Fatal error: {ex.Message}\n{ex.StackTrace}" },
                CompiledSuccessfully = false
            };
            
            Console.WriteLine(JsonSerializer.Serialize(errorResult));
            Environment.Exit(1);
        }
    }
}