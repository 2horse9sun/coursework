import sys

# Function to find Minimum Edit Distance between str1 and str2
# m and n are the number of characters in str1 and str2 respectively
def med_dp(str1, str2, m, n):

	# base case: empty strings (case 1)
	if m == 0:
		return n

	if n == 0:
		return m

	# if last characters of the strings match (case 2)
	cost = 0 if (str1[m - 1] == str2[n - 1]) else 2

	return min(med_dp(str1, str2, m - 1, n) + 1,			# deletion (case 3a))
			   med_dp(str1, str2, m, n - 1) + 1,			# insertion (case 3b))
			   med_dp(str1, str2, m - 1, n - 1) + cost)	 # substitution (case 2 + 3c)


# Memoized version of MED
def med_memoized_dp(str1, str2):

	m, n = len(str1), len(str2)

	T = [[0 for x in range(n + 1)] for y in range(m + 1)]

	for i in range(1, m + 1):
		T[i][0] = i					 # (case 1)

	for j in range(1, n + 1):
		T[0][j] = j					 # (case 1)

	for i in range(1, m + 1):
		for j in range(1, n + 1):
			if str1[i - 1] == str2[j - 1]:	# (case 2)
				cost = 0				# (case 2)
			else:
				cost = 2				# (case 3c)
			T[i][j] = min(T[i - 1][j] + 1,		  # deletion (case 3b)
						  T[i][j - 1] + 1,		  # insertion (case 3a)
						  T[i - 1][j - 1] + cost)   # replace (case 2 + 3c)

	return T[m][n], T

# Backtrack path along T
def med_backtrack(T):
	path = {}
	diff = [[-1, 0], [0, -1], [-1, -1]]
	i = len(T) - 1
	j = len(T[0]) - 1
	while i!=0 or j!=0:
		vals = [T[i+diff[0][0]][j+diff[0][1]],
				T[i+diff[1][0]][j+diff[1][1]],
				T[i+diff[2][0]][j+diff[2][1]]]
		min_val = min(vals)
		min_idx = vals.index(min_val)
		i += diff[min_idx][0]
		j += diff[min_idx][1]
		path[(i, j)] = [i-diff[min_idx][0], j-diff[min_idx][1]]
	return path

# Output each step of the shortest path
def output_steps(T, str1, str2):
	path = med_backtrack(T)
	i = 0
	j = 0
	cnt = 0
	prev = str1
	print("Steps:")
	while i!=len(T)-1 or j!=len(T[0])-1:
		next_i = path[(i, j)][0]
		next_j = path[(i, j)][1]
		if T[next_i][next_j]!=T[i][j]:
			str = str2[:next_j] + str1[next_i:]
			cnt += 1
			if next_i-i==1 and next_j-j==0:
				print('Step {} delete "{}": "{}" => "{}"'.format(cnt, str1[i], prev, str))
			if next_i-i==0 and next_j-j==1:
				print('Step {} insert "{}": "{}" => "{}"'.format(cnt, str2[j], prev, str))
			if next_i-i==1 and next_j-j==1:
				print('Step {} replace "{}" with "{}": "{}" => "{}"'.format(cnt,str1[i], str2[j], prev, str))
			prev = str
		i, j = next_i, next_j



if __name__ == '__main__':

	opt = sys.argv[1]
	str1 = sys.argv[2]
	str2 = sys.argv[3]

	if opt == '-med':
		print('Input: "{}" "{}"'.format(str1, str2))
		print("MED: {}".format(med_dp(str1, str2, len(str1), len(str2))))
	if opt == '-step':
		print('Input: "{}" "{}"'.format(str1, str2))
		med, T = med_memoized_dp(str1, str2)
		print("MED: {}".format(med))
		print("Table:\n{}".format(T))
		output_steps(T, str1, str2)



