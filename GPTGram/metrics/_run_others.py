from . import Metrics
# Open txt file "test.txt" and split the lines into 2 lists: "test_input" and "test_target" according to the separator "silvio: "
test_input = []
test_target = []
with open(test_output_file, 'r') as infile:
    lines = infile.readlines()
    for line in lines:
        # Split the line into input and target
        input, target = line.split("silvio: ")
        # Add the input and target to the lists
        test_input.append(input)
        test_target.append(target)

# Make a for on the test_input list:
metrics = Metrics()
f1_score_list = []
bleu_score_list = []
rouge_1_list = []
rouge_2_list = []
rouge_l_list = []

for line,idx in test_input:
    predictions = model.predict(line)
    #Preprocess the predictions!!
    predictions.casefold()
    metrics_dict = metrics.evaluate_metrics(predictions, test_target[idx])
    f1_score = metrics_dict["f1_score"]
    bleu_score = metrics_dict["bleu_score"]
    rouge_1 = metrics_dict["rouge_1"]
    rouge_2 = metrics_dict["rouge_2"]
    rouge_l = metrics_dict["rouge_l"]
    # Save the average of the metrics in a list
    f1_score_list.append(f1_score)  
    bleu_score_list.append(bleu_score)
    rouge_1_list.append(rouge_1)
    rouge_2_list.append(rouge_2)
    rouge_l_list.append(rouge_l)

#Take the average of the metrics in the lists:
f1_score_avg = sum(f1_score_list) / len(f1_score_list)
bleu_score_avg = sum(bleu_score_list) / len(bleu_score_list)
rouge_1_avg = sum(rouge_1_list) / len(rouge_1_list)
rouge_2_avg = sum(rouge_2_list) / len(rouge_2_list)
rouge_l_avg = sum(rouge_l_list) / len(rouge_l_list)

# Print the average of the metrics:
print(f"F1 Score: {f1_score_avg:.2f}")
print(f"BLEU Score: {bleu_score_avg:.2f}")
print(f"ROUGE-1: {rouge_1_avg:.2f}")
print(f"ROUGE-2: {rouge_2_avg:.2f}")
print(f"ROUGE-L: {rouge_l_avg:.2f}")



