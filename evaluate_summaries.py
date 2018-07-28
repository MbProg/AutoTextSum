import csv
import os
import re
from collections import defaultdict
from nltk.probability import FreqDist

eval_dir = 'evaluations/Gruppe6'
tsv_files = [f for f in os.listdir(eval_dir) if re.search(r'.tsv$', f)]

summary_names = set([s[:4] for s in tsv_files])
eval_dict = defaultdict(list)
eval_categories = []

# fill dictionary for e valuation
for sn in summary_names:
    summ_dict = defaultdict(list)
    summary_evals = [f for f in tsv_files if sn in f]
    for e in summary_evals:
        print(e)
        reader = csv.reader(open(eval_dir + '/' + e), delimiter='\t')
        rows = [l for l in reader]
        for r in rows[1:]:
            score = r[1] if r[1] is not '' else 1
            weight = r[2] if r[2] is not '' else 1
            confidence = r[3] if r[3] is not '' else 1
            summ_dict[r[0]].append((score, weight, confidence))
    eval_dict[sn] = summ_dict

# extract category names
for c in eval_dict['1001']:
    eval_categories.append(c)

# compute average score for each category
print('\naverage scores for each category \n')

for c in eval_categories:
    overall_scores = [(int(x[0])) for s in eval_dict.keys() for x in eval_dict[s][c]]
    average_overall_score = sum(overall_scores)/len(overall_scores)
    print(c + ': ' + str(average_overall_score))

# compute own score
summary_scores = []
for sn in summary_names:
    summary_score = 0
    # exclude 'Overall Quality'
    for c in eval_categories[:-1]:
        for t in eval_dict[sn][c]:
            # 5^5^5 = 625
            summary_score += int(t[0])*int(t[1])*int(t[2])
    summary_scores.append((sn, summary_score))

summary_scores_sorted = sorted(summary_scores, key=lambda x:x[1], reverse=True)

print('\naverage weighted score (mapped to [0,5])\n')
avg_own_score = sum([x[1] for x in summary_scores])/625/49
print(str(avg_own_score) + '\n')

print('\nSummaries and their weighted scores ordered from best to worst\n')
print(summary_scores_sorted)
#        print(eval_dict[sn][c])
#    for c in eval_dict[sn]:

#print('\naveraged weights\n')

for c in eval_categories:
    overall_weights = [(int(x[1])) for s in eval_dict.keys() for x in eval_dict[s][c]]
    average_overall_weights = sum(overall_weights)/len(overall_weights)
    #print(c + ': ' + str(average_overall_weights))


#print('averaged confidences\n')

for c in eval_categories:
    overall_confidences = [(int(x[2])) for s in eval_dict.keys() for x in eval_dict[s][c]]
    average_overall_confidences = sum(overall_confidences)/len(overall_confidences)
    #print(c + ': ' + str(average_overall_confidences))