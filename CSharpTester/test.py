from csharp_tester import CSharpCodeTester
import json


from collections import Counter

def calculate_error_percentages(test_results):

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

def main():
    with open("../results/gpt5_apl_to_nl_Bx_test_val.json", "r", encoding='utf-8')as f:
        data = json.load(f)
    print(len(data))
    tester = CSharpCodeTester()
    a, b = tester.test_from_json(json_path="../results/gpt5_pp_nl_to_cs_Bx_test_val.json")
    print(calculate_error_percentages(a))
    for i in a:
        for it in i:
            if (it.expected == None):
                print(i)
     
#   for idx, datapoint in enumerate(data):
#     res = tester.test_datapoint(datapoint)
#     print(idx)
#     # print(res)
#     if not all([r.passed for r in res ]):
#       print(res)
#       print("NOT PASS!!")
#     else:
#        print("passed")
       
    # if isinstance(datapoint['io'], list):
    #     failed_test = []
    #     for test in datapoint['io']:
    #         result = tester.test_single_datapoint(
    #             code=datapoint['csharp'],
    #             csharp_arg=test['CSharpArg'],
    #             expected_output=test['Output'],
    #             verbose=False
    #         )
    #         if not result.passed:
    #             failed_test.append(result)
    #             print(result)
    #     if len(failed_test)>0:
    #       print(datapoint['id'])
    # else:
    #   result = tester.test_single_datapoint(
    #       code=datapoint['csharp'],
    #       csharp_arg=datapoint['io']['CSharpArg'],
    #       expected_output=datapoint['io']['Output'],
    #       verbose=False
    #   )
    #   if not result.passed:
    #     print(datapoint['id'])
    #     print(result)


if __name__ == "__main__":
    main()
