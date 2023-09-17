import json
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

str1 = "There is enlarged effusion located at left of pleural."
str2 = "As compared to the previous radiograph, the patient has been intubated."
s1 = word_tokenize(str1)
s2 = word_tokenize(str2)
ref1 = "As compared to the previous radiograph, small left and moderate layering right pleural effusions have increased in size."
r1 = word_tokenize(ref1)

# ref2 = "Study is limited by decreased lung volumes ."
# r2 = word_tokenize(ref2)
# str1 = "As compared to the previous radiograph, the lung volumes have decreased."
# str2 = "There is decrease volume located at lung. "
# s1 = word_tokenize(str1)
# s2 = word_tokenize(str2)

weights=(1,0)

print(sentence_bleu([r1], s1,weights))
print(sentence_bleu([r1], s2,weights))


