from tabulate import tabulate


def predict(question, questions):
    import pickle
    from _csv import reader

    import pandas as pd
    from . import engineering

    def load_csv(fileN):
        dataset = list()
        with open(r'D:\papers\Project Material\new_try\CQA\forum\revised.csv', 'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
        return dataset

    def predict(node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return predict(node['right'], row)
            else:
                return node['right']

    def bagging_predict(trees, row):
        results = [predict(tree, row) for tree in trees]
        print(results)
        return max(set(results), key=results.count)

    # q = str(input("Enter the question 	:	"))
    test_id = list()
    for i in range(1, len(questions) + 1):
        test_id.append(i)
    question2 = list(questions)
    for i in range(len(questions)):
        question2[i] = question
    question_list = {
        'test_id': test_id,
        'question1': questions,
        'question2': question2,
    }
    # data = pd.read_csv(r'D:\papers\Project Material\new_try\CQA\forum\test-20.csv')
    data = pd.DataFrame(question_list, columns=['test_id', 'question1', 'question2'])
    print("Pairing questions")
    for i in range(len(data)):
        data['question2'] = question
    question1 = list(data['question1'])
    question2 = list(data['question2'])
    print("Paired: ")
    q, main_l = list(), list()
    for q1, q2 in zip(question1, question2):
        q.append(q1)
        q.append(q2)
        main_l.append(q)
        q = list()
    print(tabulate(main_l))

    print(f"Predict: Data-> {data}")

    print("Preprocessing the Question Pairs")
    engineering.engineer(data)
    print("Preprocessing finished")

    filename = 'D:\papers\Project Material\new_try\CQA\forum\\revised.csv'
    test = load_csv(filename)
    test.pop(0)
    print("Actual Prediction Started: ")
    with open(r'D:\papers\Project Material\new_try\CQA\forum\trees_8000.pickle', 'rb') as f:
        trees = pickle.load(f)
    predictions = [bagging_predict(trees, row) for row in test]
    i = 0
    print("Prediction Complete")

    lst = list()
    main_list = list()
    for q1, q2 in zip(question1, question2):
        lst.append(i)
        lst.append(q1)
        lst.append(q2)
        lst.append(predictions[i])
        main_list.append(lst)
        lst = list()
        i += 1

    print(tabulate(main_list))

    return predictions, main_list
