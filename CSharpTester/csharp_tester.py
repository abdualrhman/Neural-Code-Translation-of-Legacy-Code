from collections import Counter
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SingleTestResult:
    passed: bool
    expected: Any
    actual: Any
    csharp_args: Any
    error_message: Optional[str] = None
    error_class: Optional[str] = None
    def to_dict(self):
        return {
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual,
            "csharp_args": self.csharp_args,
            "error_message": self.error_message,
            "error_class": self.error_class,
        }
    def __str__(self):
        if self.passed:
            return f"✓ PASSED: Expected {self.expected}, got {self.actual}"
        else:
            if self.error_message:
                return f"✗ FAILED: {self.error_message}"
            else:
                return f"✗ FAILED: Expected {self.expected}, got {self.actual}"


@dataclass
class TestResult:
    TotalTests: int
    PassedTests: int
    FailedTests: int
    FailureMessages: List[str]
    AllPassed: bool
    CompiledSuccessfully: bool
    individual_results: List[SingleTestResult] = None
    def __post_init__(self):
        if self.individual_results is None:
            self.individual_results = []


@dataclass
class AggregateTestStats:
    TotalSubmissions: int
    CompiledSubmissions: int
    PartialPassSubmissions: int
    FullPassSubmissions: int
    CompileRate: float
    PartialPassRate: float
    FullPassRate: float


class CSharpCodeTester:
    def __init__(self, csharp_executor_path: Optional[str] = None):
        if csharp_executor_path is None:
            self.executor_path, self.is_self_contained = self._find_executor()
        else:
            self.executor_path = csharp_executor_path
            self.is_self_contained = not csharp_executor_path.endswith('.dll')
    
    @staticmethod
    def _find_executor():
        """Find the C# executor in common locations."""
        possible_paths = [
            # Self-contained executables (no .NET runtime needed)
            Path("CSharpTester"), 
            Path("CSharpTester.exe"),  
            Path("CSharpTester/bin/Release/net9.0/linux-x64/publish/CSharpTester"),
            Path("CSharpTester/bin/Release/net9.0/win-x64/publish/CSharpTester.exe"),
            Path("CSharpTester/bin/Release/net9.0/osx-x64/publish/CSharpTester"),
            # DLL versions (require .NET runtime)
            Path("CSharpTester.dll"),
            Path("CSharpTester/bin/Release/net9.0/CSharpTester.dll"),
            Path("CSharpTester/bin/Debug/net9.0/CSharpTester.dll"),
            Path("bin/Release/net9.0/CSharpTester.dll"),
            Path("bin/Debug/net9.0/CSharpTester.dll"),
        ]
        
        for path in possible_paths:
            if path.exists():
                is_self_contained = not str(path).endswith('.dll')
                return str(path), is_self_contained
        
        raise FileNotFoundError(
            "Could not find CSharpTester.dll or CSharpTester executable. "
            "Please specify the path or build the C# project."
        )
    
    @staticmethod
    def _decode_csharp_string(code: str) -> str:
        """Decode escaped characters in C# code strings."""
        if not any(seq in code for seq in ['\\n', '\\/', '\\t', '\\r', '\\\\']):
            return code
        
        try:
            # Protect sequences that should remain escaped
            decoded = code.replace('\\\\', '\x00')  # Placeholder for \\
            decoded = decoded.replace("\\'", '\x01')  # Placeholder for \'
            decoded = decoded.replace('\\"', '\x02')  # Placeholder for \"
            
            # Decode sequences that should be converted
            decoded = decoded.replace('\\n', '\n')
            decoded = decoded.replace('\\t', '\t')
            decoded = decoded.replace('\\r', '\r')
            
            # Restore protected sequences
            decoded = decoded.replace('\x00', '\\')   # \\ -> \
            decoded = decoded.replace('\x01', "'")    # \' -> '
            decoded = decoded.replace('\x02', '"')    # \" -> "
            
            return decoded
        except Exception:
            return code
    
    def _parse_individual_results(self, result_data: Dict, test_cases: List[Dict]) -> List[SingleTestResult]:
        """Parse individual test results from C# executor output."""
        individual_results = []
        
        for i, test_case in enumerate(test_cases):
            test_num = i + 1
            failure_msg = next(
                (msg for msg in result_data['FailureMessages'] 
                 if msg.startswith(f"  Test {test_num}:")),
                None
            )
            
            if failure_msg:
                # Test failed
                if "Expected" in failure_msg and "got" in failure_msg:
                    parts = failure_msg.split("Expected ", 1)[1].split(", got ")
                    expected_str = parts[0]
                    actual_str = parts[1] if len(parts) > 1 else "unknown"
                    individual_results.append(SingleTestResult(
                        passed=False,
                        expected=test_case['Output'],
                        actual=actual_str,
                        csharp_args=test_case['CSharpArg'],
                        error_message=None,
                        error_class="functional"
                    ))
                elif "Compilation failed" in failure_msg:
                    error_text = failure_msg.split(": ", 1)[1] if ": " in failure_msg else failure_msg
                    individual_results.append(SingleTestResult(
                        passed=False,
                        expected=test_case['Output'],
                        actual=None,
                        csharp_args=test_case['CSharpArg'],
                        error_message=error_text,
                        error_class="compile"
                    ))

                else:
                    # Other error
                    error_text = failure_msg.split(": ", 1)[1] if ": " in failure_msg else failure_msg
                    individual_results.append(SingleTestResult(
                        passed=False,
                        expected=test_case['Output'],
                        actual=None,
                        csharp_args=test_case['CSharpArg'],
                        error_message=error_text,
                        error_class="runtime"
                    ))
            else:
                individual_results.append(SingleTestResult(
                    passed=True,
                    expected=test_case['Output'],
                    actual=test_case['Output'],
                    csharp_args=test_case['CSharpArg']
                ))
        
        return individual_results
    
    def test_code(self, code: str, test_cases: List[Dict[str, Any]]) -> TestResult:
        # code = self._decode_csharp_string(code)
        
        input_data = {
            "code": code,
            "testCases": test_cases
        }
        input_json = json.dumps(input_data)
        
        try:
            cmd = [self.executor_path] if self.is_self_contained else ["dotnet", self.executor_path]
            
            result = subprocess.run(
                cmd,
                input=input_json,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                print("======--------")
                print(result.stderr)
                return TestResult(
                    TotalTests=len(test_cases),
                    PassedTests=0,
                    FailedTests=len(test_cases),
                    FailureMessages=[f"Execution error: {result.stderr}"],
                    AllPassed=False,
                    CompiledSuccessfully=False,
                    individual_results=[],

                )
            
            result_data = json.loads(result.stdout)
            test_result = TestResult(
                TotalTests=result_data['TotalTests'],
                PassedTests=result_data['PassedTests'],
                FailedTests=result_data['FailedTests'],
                FailureMessages=result_data['FailureMessages'],
                AllPassed=result_data['AllPassed'],
                CompiledSuccessfully=result_data['CompiledSuccessfully']
            )
            
            if result_data['CompiledSuccessfully']:
                test_result.individual_results = self._parse_individual_results(result_data, test_cases)
            
            return test_result

        except subprocess.TimeoutExpired:
            return TestResult(
                TotalTests=len(test_cases),
                PassedTests=0,
                FailedTests=len(test_cases),
                FailureMessages=["Execution timeout (30s exceeded)"],
                AllPassed=False,
                CompiledSuccessfully=False,
                individual_results=[]
            )
        except json.JSONDecodeError as e:

            return TestResult(
                TotalTests=len(test_cases),
                PassedTests=0,
                FailedTests=len(test_cases),
                FailureMessages=[f"Failed to parse result: {e}\nOutput: {result.stdout}"],
                AllPassed=False,
                CompiledSuccessfully=False,
                individual_results=[]
            )
        except Exception as e:
            return TestResult(
                TotalTests=len(test_cases),
                PassedTests=0,
                FailedTests=len(test_cases),
                FailureMessages=[f"Unexpected error: {str(e)}"],
                AllPassed=False,
                CompiledSuccessfully=False,
                individual_results=[]
            )
    
    def test_datapoint(self, datapoint: Dict[str, Any]) -> Optional[List[SingleTestResult]]:
        io_data = datapoint.get('io')
        
        if not io_data:
            return None
        
        test_cases = io_data if isinstance(io_data, list) else [io_data]
        
        result = self.test_code(datapoint['model_pred'], test_cases)
        
        if result.individual_results:
            return result.individual_results
        
        if not result.CompiledSuccessfully:
            error_msg = '; '.join(result.FailureMessages)
            return [
                SingleTestResult(
                    passed=False,
                    expected=test['Output'],
                    actual=None,
                    csharp_args=test['CSharpArg'],
                    error_message=f"Compilation failed: {error_msg}",
                    error_class="compile"
                )
                for test in test_cases
            ]
        
        return [
            SingleTestResult(
                passed=result.AllPassed,
                expected=test['Output'],
                actual=test['Output'] if result.AllPassed else None,
                csharp_args=test['CSharpArg'],
                error_message=None if result.AllPassed else "Test failed"
            )
            for test in test_cases
        ]
    
    def calculate_stats(self, results: List[TestResult]) -> AggregateTestStats:
        total = len(results)
        for i in results:
            if not i.CompiledSuccessfully:
                print("------")
                print(i)
        compiled = sum(1 for r in results if r.CompiledSuccessfully)
        partial_pass = sum(1 for r in results if r.CompiledSuccessfully and r.PassedTests > 0)
        full_pass = sum(1 for r in results if r.CompiledSuccessfully and r.AllPassed)
        
        compile_rate = (compiled / total * 100) if total > 0 else 0
        partial_rate = (partial_pass / compiled * 100) if compiled > 0 else 0
        full_rate = (full_pass / compiled * 100) if compiled > 0 else 0
        
        return AggregateTestStats(
            TotalSubmissions=total,
            CompiledSubmissions=compiled,
            PartialPassSubmissions=partial_pass,
            FullPassSubmissions=full_pass,
            CompileRate=compile_rate,
            PartialPassRate=partial_rate,
            FullPassRate=full_rate
        )
    
    def print_summary(self, stats: AggregateTestStats):
        """Print summary statistics."""
        print("=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print("\nCompilation Statistics:")
        print(f"  Compile Rate: {stats.CompileRate:.2f}% "
              f"({stats.CompiledSubmissions}/{stats.TotalSubmissions})")
        
        print("\nPass Rate Statistics (of compiled code):")
        print(f"  Partial Pass Rate: {stats.PartialPassRate:.2f}% "
              f"({stats.PartialPassSubmissions}/{stats.CompiledSubmissions})")
        print(f"  Full Pass Rate: {stats.FullPassRate:.2f}% "
              f"({stats.FullPassSubmissions}/{stats.CompiledSubmissions})")
        
        if stats.FullPassSubmissions == stats.TotalSubmissions:
            print("\n🎉 All tests passed!")
        else:
            failed = stats.TotalSubmissions - stats.FullPassSubmissions
            print(f"\n⚠️  {failed} code block(s) failed")
    

    def calculate_error_percentages(self, test_results):

        all_results = [result for test_case in test_results for result in test_case]

        error_classes = [result.error_class for result in all_results]
        error_counts = Counter(error_classes)
        print(error_classes)
        print(error_counts)
        total = len(error_classes)

        percentages = {
            error_class: (count / total) * 100 
            for error_class, count in error_counts.items()
        }

        return percentages
    
    
    def test_from_json(
        self, 
        json_path: str, 
        verbose: bool = False,
        print_stats: bool = True
    ) -> tuple[List[List[SingleTestResult]], AggregateTestStats]:
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            datapoints = json.load(f)
        
        if not isinstance(datapoints, list):
            raise ValueError("JSON file must contain a list of datapoints")
        
        print(f"Loading {len(datapoints)} datapoints from {json_path}...")
        
        all_results = []
        test_results_for_stats = []
        
        for idx, datapoint in enumerate(datapoints):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Testing datapoint {idx + 1}/{len(datapoints)}")
                print(f"{'='*50}")
            
            single_results = self.test_datapoint(datapoint)
            
            if single_results is None:
                if verbose:
                    print("⚠️  Skipped (no test cases)")
                continue
            
            all_results.append(single_results)
            
            passed_tests = sum(1 for r in single_results if r.passed)
            failed_tests = len(single_results) - passed_tests
            compiled = not any(r.error_message and "Compilation failed" in r.error_message 
                             for r in single_results)
            
            failure_messages = [
                r.error_message or f"Expected {r.expected}, got {r.actual}"
                for r in single_results if not r.passed
            ]
            
            test_result = TestResult(
                TotalTests=len(single_results),
                PassedTests=passed_tests,
                FailedTests=failed_tests,
                FailureMessages=failure_messages,
                AllPassed=passed_tests == len(single_results),
                CompiledSuccessfully=compiled,
                individual_results=single_results
            )
            test_results_for_stats.append(test_result)
            
            if verbose:
                for result in single_results:
                    print(result)
        
        if test_results_for_stats:
            stats = self.calculate_stats(test_results_for_stats)
            if print_stats:
                print(f"\n")
                self.print_summary(stats)
        else:
            print("\n⚠️  No valid test cases found in JSON file")
            stats = AggregateTestStats(
                TotalSubmissions=0,
                CompiledSubmissions=0,
                PartialPassSubmissions=0,
                FullPassSubmissions=0,
                CompileRate=0.0,
                PartialPassRate=0.0,
                FullPassRate=0.0
            )
        
        return all_results, stats