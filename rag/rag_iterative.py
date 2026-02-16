from dataclasses import dataclass
import json
import os
import re
from typing import List, Dict, Any, Tuple
import sys
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core import SimpleDirectoryReader,load_index_from_storage,StorageContext
from llama_index.llms.azure_openai import AzureOpenAI as EmbedAzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from openai import AzureOpenAI

load_dotenv()

parent_dir = Path(__file__).parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
from CSharpTester.csharp_tester import CSharpCodeTester


@dataclass
class TranslationConfig:
    input_path: str
    output_path: str
    persist_dir: str = "./apl_index"
    docs_dir: str = "./apl_docs"
    similarity_top_k: int = 5
    max_retry_attempts: bool = 1
    debug_translation: bool = False

    output_llm_api_key: str = None
    output_llm_deployment: str = None
    output_llm_api_version: str = None
    output_llm_endpint: str = None

    rag_llm_deployment: str = None
    rag_llm_api_version: str = None
    rag_llm_endpoint: str = None
    rag_llm_api_key: str = None
    
    embedding_model_deployment: str = None
    embedding_model_api_version: str = None
    embedding_model_api_key: str = None
    embedding_model_endpoint: str = None

def extract_csharp(text: str) -> str:

    md_pattern = re.compile(
        r"```(?:csharp|C#)?\s*(.*?)```",
        re.DOTALL
    )
    md_blocks = md_pattern.findall(text)

    if md_blocks:
        last_block = md_blocks[-1].strip()
        return last_block

    code_start_keywords = [
        "using ",
        "namespace ",
        "class ",
        "public ",
        "internal ",
        "static "
    ]

    starts = []
    for kw in code_start_keywords:
        idx = text.rfind(kw)
        if idx != -1:
            starts.append(idx)

    if not starts:
        return text.strip()

    start = max(starts)
    candidate = text[start:]

    brace_count = 0
    end_index = None

    for i, ch in enumerate(candidate):
        if ch == "{":
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0:
                end_index = i

    if end_index is not None:
        cleaned = candidate[:end_index + 1]
        return cleaned.strip()

    return candidate.strip()


class IterativeRAGPipeline:
    def __init__(self, config: TranslationConfig):
        self.config = config
        self._setup_llm()
        self._setup_output_llm()
        self._setup_index()
        self.tester = CSharpCodeTester("../CSharpTester/bin/Release/net9.0/osx-x64/CSharpTester")
        self.attempt_test_results = {}

    def _setup_output_llm(self):
        self.output_llm_client = AzureOpenAI(
            api_version=self.config.output_llm_api_version,
            azure_endpoint=self.config.output_llm_endpint,
            api_key=self.config.output_llm_api_key,
        )

    def _setup_llm(self):
        Settings.llm = EmbedAzureOpenAI(
            model=self.config.rag_llm_deployment,
            deployment_name=self.config.rag_llm_deployment,
            api_key=self.config.rag_llm_api_key,
            azure_endpoint=self.config.rag_llm_endpoint,
            api_version=self.config.rag_llm_api_version,
        )
        
        Settings.embed_model = AzureOpenAIEmbedding(
            model=self.config.embedding_model_deployment,
            deployment_name=self.config.embedding_model_deployment,
            api_key=self.config.embedding_model_api_key,
            azure_endpoint=self.config.embedding_model_endpoint,
            api_version=self.config.embedding_model_api_version,
        )
    
    def _setup_index(self):
        if os.path.exists(self.config.persist_dir):
            storage_context = StorageContext.from_defaults(persist_dir=self.config.persist_dir)
            self.index = load_index_from_storage(storage_context)
        else:
            documents = SimpleDirectoryReader(self.config.docs_dir).load_data()
            self.index = VectorStoreIndex.from_documents(documents)
            self.index.storage_context.persist(persist_dir=self.config.persist_dir)
        
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=self.config.similarity_top_k
        )
    

    def _build_messages(
        self, 
        apl_code:str,
        method_signatures: str,
        context,
        previous_attempts: List[Tuple[str, List]] = None
    ) -> list[dict]:
        system_content ="""You are an expert APL code programmer.\n
            Given the following APL code create C# program that implements the given code.\n"""
        
        user_content = f"### APL code:\n{apl_code}\n"
        
        if method_signatures:
            user_content += (
                "\n### Required C# method signatures (overloads)\n"
                "You MUST implement all of the following overloads exactly as written "
                "(names, return types, and parameter types):\n"
                f"{method_signatures}\n"
            )
        # if context:
        #     user_content += f"### Context: \n{context}"
        if previous_attempts:
            user_content += "\n### Previous Attempts and Test Failures\n"
            user_content += "The following code attempts failed tests. Learn from these errors:\n\n"
            
            for attempt_num, (prev_code, test_results) in enumerate(previous_attempts, 1):
                user_content += f"#### Attempt {attempt_num}:\n"
                user_content += f"```csharp\n{prev_code}\n```\n\n"
                user_content += "**Test Results:**\n"
                
                for i, result in enumerate(test_results, 1):
                    user_content += f"Test {i}:\n"
                    user_content += f"  - Passed: {result.passed}\n"
                    if not result.passed:
                        user_content += f"  - Error: {result.error_message}\n"
                        user_content += f"  - CSharp arguments: {result.csharp_args}\n"
                    if hasattr(result, 'expected_output') and result.expected_output:
                        user_content += f"  - Expected: {result.expected_output}\n"
                    if hasattr(result, 'actual_output') and result.actual_output:
                        user_content += f"  - Actual: {result.actual_output}\n"
                user_content += "\n"
            
            user_content += "Please fix the issues and generate corrected C# code.\n"
        
        user_content += "Output format: Only compilable C# program code, no explanations, no reasoning, no example usage."
        user_content += "### C#:"
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        if self.config.debug_translation:
            self._print_debug_info(apl_code, context)
        return messages

    def _print_debug_info(self, apl_code, context):
        print("=" * 50)
        print("APL CODE")
        print(apl_code)
        print("=" * 50)
        print("CONTEXT")
        print(context)
        print("\nSOURCES:")
        for i, node in enumerate(context.source_nodes):
            print(f"\n--- source {i} ---")
            print("score:", getattr(node, "score", None))
            print("metadata:", node.node.metadata)
            print(node.node.get_content())
        print("=" * 50)

    def generate_code(
        self, 
        apl_code: str, 
        method_signatures: str,
        previous_attempts: List[Tuple[str, List]] = None
    ) -> str:
        """Generate C# code from natural language description."""

        context = self.query_engine.query(apl_code)
        messages = self._build_messages(apl_code, method_signatures, context, previous_attempts)
        
        response = self.output_llm_client.chat.completions.create(
            messages=messages,
            max_tokens=16384,
            model=self.config.output_llm_deployment
        )
        
        text = getattr(response, "output_text", None)
        if not text:
            try:
                text = response.choices[0].message.content
            except Exception:
                text = str(response)
        
        return text.strip()


    def generate_with_testing(
        self,
        apl_code: str,
        method_signatures: str,
        test_io_data: List[Dict],
        max_attempts: int = 1
    ) -> Tuple[str, int, bool]:

        previous_attempts = []
        
        for attempt in range(1, max_attempts + 1):
            print(f"  Attempt {attempt}/{max_attempts}")

            raw_output = self.generate_code(
                apl_code, 
                method_signatures,
                previous_attempts if previous_attempts else None
            )
            
            print(f"Raw output preview: {raw_output[:200]}...")
            generated_code = extract_csharp(raw_output)
            
            test_datapoint = {
                "method_signatures": method_signatures,
                "model_pred": generated_code,
                "io": test_io_data
            }
            try:
                test_results = self.tester.test_datapoint(datapoint=test_datapoint)
                attempt_key = str(attempt)
                if attempt_key not in self.attempt_test_results:
                    self.attempt_test_results[attempt_key] = []
                self.attempt_test_results[attempt_key].append(test_results)

                all_passed = all(result.passed for result in test_results)
                if all_passed:
                    print(f"  ✓ All tests passed on attempt {attempt}")
                    return generated_code, attempt, True
                else:
                    failed_count = sum(1 for r in test_results if not r.passed)
                    print(f"  ✗ {failed_count}/{len(test_results)} tests failed")
                    previous_attempts.append((generated_code, test_results))
                    
            except Exception as e:
                print(f"  Error during testing: {e}")
                # Store attempt with error
                previous_attempts.append((generated_code, []))
        
        print(f"  Max attempts ({max_attempts}) reached without passing all tests")
        return generated_code, max_attempts, False

    def evaluate_dataset(self,):
        with open(self.config.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results: List[Dict[str, Any]] = []

        for i, item in enumerate(data):
            print(f"\nProcessing item {i+1}/{len(data)}")
            
            if 'io' not in item or not item['io']:
                print(f"  Skipping item {i+1}: no test cases")
                results.append({
                    **item,
                    "model_pred": "",
                    "test_attempts": 0,
                    "passed_all_tests": False
                })
                continue
            try:
                generated_code, attempts, passed = self.generate_with_testing(
                    apl_code=item['apl'],
                    method_signatures=item.get('method_signatures', ''),
                    test_io_data=item['io'],
                    max_attempts=self.config.max_retry_attempts
                )
                
                results.append({
                    **item,
                    "model_pred": generated_code,
                    "test_attempts": attempts,
                    "passed_all_tests": passed
                })
                print("✓ Completed")
            except Exception as e:
                print(f"✗ Error: {e}")
                results.append({
                    **item,
                    "model_pred": "",
                    "test_attempts": self.config.max_retry_attempts,
                    "passed_all_tests": False
                })

        with open(self.config.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("Error class distribution")
        for attempt in self.attempt_test_results:
            all_results = [result for test_case in self.attempt_test_results[attempt] for result in test_case]
            error_classes = [result.error_class for result in all_results]
            total = self.count_ios(data)
            dif = total - len(error_classes)
            error_classes.extend([None]* dif)
            self.attempt_test_results[attempt] = error_classes

        print(self.attempt_test_results)

        
        return results
    def count_ios(self, data):
        total = 0
        for item in data:
            io = item.get('io')
            if isinstance(io, list):
                total += len(io)
            elif isinstance(io, dict):
                total += 1
        return total

def main():
    PERSIST_DIR = "./apl_index"
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
    API_KEY= os.getenv("SUBSCRIPTION_KEY")
    
    OUTPUT_LLM_MODEL_DEPLOYMENT = "gpt-5-chat"
    OUTPUT_LLM_API_VERSION = "2024-12-01-preview"

    RAG_LLM_DEPLOYMENT = "gpt-5-mini"
    RAG_LLM_DEPLOYMENT = "gpt-5-mini"
    RAG_LLM_API_VERSION = "2024-12-01-preview"

    EMBEDDING_MODEL_DEPLOYMENT = "text-embedding-3-large"
    EMBEDDING_MODEL_API_VERSION = "2024-12-01-preview"
    rag_config = TranslationConfig(
        input_path="../data/Bx_test_val.json",
        output_path="../results/gpt5_rag_Bx_test_val_iterative.json",
        persist_dir=PERSIST_DIR,
        max_retry_attempts=5,
        debug_translation=False,
        ## azure cred ## 
        # output llm 
        output_llm_api_key=API_KEY,
        output_llm_deployment = OUTPUT_LLM_MODEL_DEPLOYMENT,
        output_llm_api_version = OUTPUT_LLM_API_VERSION,
        output_llm_endpint = AZURE_ENDPOINT,
        # rag llm
        rag_llm_deployment=RAG_LLM_DEPLOYMENT,
        rag_llm_api_version=RAG_LLM_API_VERSION,
        rag_llm_api_key=API_KEY,
        rag_llm_endpoint=AZURE_ENDPOINT,
        # rag embedding model
        embedding_model_deployment=EMBEDDING_MODEL_DEPLOYMENT,
        embedding_model_api_version=EMBEDDING_MODEL_API_VERSION,
        embedding_model_api_key=API_KEY,
        embedding_model_endpoint=AZURE_ENDPOINT
    )
    pipeline = IterativeRAGPipeline(rag_config)
    pipeline.evaluate_dataset()


if __name__ == "__main__":
    main()