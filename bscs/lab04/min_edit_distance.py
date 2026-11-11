def D(i, j):
    if i == 0:
        return j
    if j == 0:
        return i
    if word1[i - 1] == word2[j - 1]:
        return D(i - 1, j - 1)
    term1 = D(i - 1, j) + 1
    term2 = D(i, j - 1) + 1
    term3 = D(i - 1, j - 1) + 2 if word1[i - 1] != word2[j - 1] else D(i - 1, j - 1)
    return min(term1, term2, term3)

def ptr(i, j):
    if i == 0:
        return 'I' * j
    if j == 0:
        return 'D' * i
    if word1[i - 1] == word2[j - 1]:
        return ptr(i - 1, j - 1) + 'M'
    term1 = D(i - 1, j) + 1
    term2 = D(i, j - 1) + 1
    term3 = D(i - 1, j - 1) + 2 if word1[i - 1] != word2[j - 1] else D(i - 1, j - 1)
    if term3 <= term1 and term3 <= term2:
        return ptr(i - 1, j - 1) + ('S' if word1[i - 1] != word2[j - 1] else 'M')
    elif term2 < term3 and term2 < term1:
        return ptr(i, j - 1) + 'I'
    else:
        return ptr(i - 1, j) + 'D'

word1 = "execution"
word2 = "intention"
print(f"Minimum edit distance between '{word1}' and '{word2}': {D(len(word1), len(word2))}")
print(f"Edit sequence: {ptr(len(word1), len(word2))}")