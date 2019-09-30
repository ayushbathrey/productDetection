# Pythono3 code to rename multiple 
# files in a directory or folder 

# importing os module 
import os 

# Function to rename multiple files 
def main(): 
	i = 0
	dir="test/"
	for filename in os.listdir(dir): 
		dst ="Image" + str(i) + ".jpg"
		src = dir + filename 
		dst = dir + dst 
		
		# rename() function will 
		# rename all the files 
		os.rename(src, dst) 
		i += 1

# Driver Code 
if __name__ == '__main__': 
	
	# Calling main() function 
	main() 
