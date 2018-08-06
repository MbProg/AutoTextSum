import csv
import os
import re
import scipy
import matplotlib
import pandas
from collections import defaultdict
from collections import OrderedDict
from nltk.probability import FreqDist

eval_dir = 'evaluations/Baseline2'
tsv_files = [f for f in os.listdir(eval_dir) if re.search(r'.tsv$', f)]

summary_names = set([s[:4] for s in tsv_files])
eval_dict = defaultdict(list)
eval_categories = []

eval_dict_per_anno = defaultdict(list)

# fill dictionary for e valuation
for sn in summary_names:
    summ_dict = defaultdict(list)
    summary_evals = [f for f in tsv_files if sn in f]
    for e in summary_evals:
        #print(e)
        reader = csv.reader(open(eval_dir + '/' + e), delimiter='\t')
        rows = [l for l in reader]
        for r in rows[1:]:
            score = r[1] if r[1] is not '' else 1
            weight = r[2] if r[2] is not '' else 1
            confidence = r[3] if r[3] is not '' else 1
            summ_dict[r[0]].append((score, weight, confidence))
            # print comments of a particular evaluation category
            #if r[0]=='Structure' and int(score) < 3:
                #print(e)
                #print(r[4])
    eval_dict[sn] = summ_dict

# fill dictionary per annotation
for sn in summary_names:
    summary_evals = [f for f in tsv_files if sn in f]
    annos = []
    for e in summary_evals:
        anno = []
        #print(e)
        reader = csv.reader(open(eval_dir + '/' + e), delimiter='\t')
        rows = [l for l in reader]
        for r in rows[1:]:
            score = r[1] if r[1] is not '' else 1
            weight = r[2] if r[2] is not '' else 1
            confidence = r[3] if r[3] is not '' else 1
            anno.append((score, weight, confidence))
            # print comments of a particular evaluation category
            #if r[0]=='Structure' and int(score) < 3:
                #print(e)
                #print(r[4])
        annos.append(anno)
    eval_dict_per_anno[sn] = annos
# extract category names
for c in eval_dict['1001']:
    eval_categories.append(c)

# compute average score for each category
#print('\naverage scores for each category \n')

#for c in eval_categories:
#    overall_scores = [(int(x[0])) for s in eval_dict.keys() for x in eval_dict[s][c]]
#    average_overall_score = sum(overall_scores)/len(overall_scores)
    #print(c + ': ' + str(average_overall_score))


# compute own score
#summary_scores = []
weighted_scores = []
weighted_scores_with_names = []

for sn in summary_names:
    for a in eval_dict_per_anno[sn]:
        weighted_score = 0
        for t in a:
            weighted_score += (int(t[0])*int(t[1])*int(t[2]))/160
        weighted_scores.append(weighted_score)
        weighted_scores_with_names.append((sn,weighted_score))


#print(weighted_scores_with_names)
#print(weighted_scores)


# create lists for correlation calculation

overall_scores = []
#grammaticality_scores = []
#non_redundancy_scores = []
#referential_clarity_scores = []
#focus_scores = []
#structure_scores = []
#coherence_scores = []
#readability_scores = []
#information_content_scores = []
#spelling_scores = []
#length_scores = []

# TODO: here the work is done

overall_scores_with_names = []

for sn in summary_names:
    #overall_scores.append(eval_dict[sn]['Overall Quality'][0])
    #print(sn)
    scores = 0
    tuples = eval_dict[sn]['Overall Quality']
    for t in tuples:
        #print(t)
        #overall_scores.append(int(t[0]))
        #overall_scores_with_names.append((int(sn), int(t[0])))
        scores += int(t[0])
    overall_scores_with_names.append((sn, scores/len(tuples)))

print(overall_scores_with_names)
print()

overall_scores_with_names.sort(key=lambda x:x[0])

#for x in overall_scores_with_names:
#    print(str(x[0]))

print("\n\n\n")

for x in overall_scores_with_names:
    print(str(x[1]))



#    for t in eval_dict[sn]['Grammaticality']:
#        grammaticality_scores.append(int(t[0]))
#    for t in eval_dict[sn]['Non-Redundancy']:
#        non_redundancy_scores.append(int(t[0]))
#    for t in eval_dict[sn]['Referential Clarity']:
#        referential_clarity_scores.append(int(t[0]))
#    for t in eval_dict[sn]['Focus']:
#        focus_scores.append(int(t[0]))
#    for t in eval_dict[sn]['Structure']:
#        structure_scores.append(int(t[0]))
#    for t in eval_dict[sn]['Coherence']:
#        coherence_scores.append(int(t[0]))
#    for t in eval_dict[sn]['Readability']:
#        readability_scores.append(int(t[0]))
#    for t in eval_dict[sn]['Information Content']:
#        information_content_scores.append(int(t[0]))
#    for t in eval_dict[sn]['Spelling']:
#        spelling_scores.append(int(t[0]))
#    for t in eval_dict[sn]['Length']:
#        length_scores.append(int(t[0]))



# Pearson's r between overall quality and weighted score
#print('\n------------ Pearsons r overall quality - own score --------------------------------\n')
#pearson_correlation_overall_own = scipy.stats.stats.pearsonr(overall_scores, weighted_scores)
#print(str(pearson_correlation_overall_own) + '\n')
#print('funktioniert NICHT, da Listen in unterschiedlicher Summaryreihenfolge!!!')

#category_scores_for_ic = [("grammaticality_scores", grammaticality_scores),
#("non_redundancy_scores", non_redundancy_scores),
#("referential_clarity_scores", referential_clarity_scores),
#("focus_scores", focus_scores),
#("structure_scores", structure_scores),
#("coherence_scores", coherence_scores),
#("readability_scores", readability_scores),
#("spelling_scores", spelling_scores),
#("length_scores", length_scores)]

#print("Correlation with information content: ")
#print([(n, str(scipy.stats.stats.pearsonr(l, information_content_scores))) for n, l in category_scores_for_ic])


#category_scores_for_g = [("grammaticality_scores", grammaticality_scores),
#("non_redundancy_scores", non_redundancy_scores),
#("referential_clarity_scores", referential_clarity_scores),
#("focus_scores", focus_scores),
#("structure_scores", structure_scores),
#("coherence_scores", coherence_scores),
#("information_content_scores", information_content_scores),
#("spelling_scores", spelling_scores),
#("length_scores", length_scores)]

#print("\nCorrelation with readability: ")
#print([(n, str(scipy.stats.stats.pearsonr(l, readability_scores))) for n, l in category_scores_for_g])

#category_scores_for_s = [("grammaticality_scores", grammaticality_scores),
#("non_redundancy_scores", non_redundancy_scores),
#("referential_clarity_scores", referential_clarity_scores),
#("focus_scores", focus_scores),
#("readability_scores", readability_scores),
#("coherence_scores", coherence_scores),
#("information_content_scores", information_content_scores),
#("spelling_scores", spelling_scores),
#("length_scores", length_scores)]

#print("\nCorrelation with structure: ")
#print([(n, str(scipy.stats.stats.pearsonr(l, structure_scores))) for n, l in category_scores_for_s])


#print('\n------------ Pearsons r informa - own score' + category + ' --------------------------------\n')
#pearson_correlation_overall_own = scipy.stats.stats.pearsonr(overall_scores, summary_scores_for_corr)
#print(str(pearson_correlation_overall_own) + '\n')


#weighted_scores_sorted = sorted(weighted_scores, key=lambda x: x[0], reverse=True)

#print('\naverage weighted score (mapped to [0,5])\n')
# TODO: avg_own_score = sum([x[1] for x in weighted_scores])/625/49
#print(str(avg_own_score) + '\n')

#print('\nSummaries and their weighted scores ordered from best to worst\n')
#print(summary_scores_sorted)
#        print(eval_dict[sn][c])
#    for c in eval_dict[sn]:

#print('\naveraged weights\n')

for c in eval_categories:
    overall_weights = [(int(x[1])) for s in eval_dict.keys() for x in eval_dict[s][c]]
    average_overall_weights = sum(overall_weights)/len(overall_weights)
#    print(c + ': ' + str(average_overall_weights))


#print('averaged confidences\n')

for c in eval_categories:
    overall_confidences = [(int(x[2])) for s in eval_dict.keys() for x in eval_dict[s][c]]
    average_overall_confidences = sum(overall_confidences)/len(overall_confidences)
    #print(c + ': ' + str(average_overall_confidences))
