from shopping import *


#####
print("test process_line()")

test_case_1 = ['0','0','0','0','1','0','0.2','0.2','0','0','Feb','1','1','1','1','Returning_Visitor','FALSE','FALSE']

correct_row = [0, 0.0, 0, 0.0, 1, 0.0, 0.2, 0.2, 0.0, 0.0, 1, 1, 1, 1, 1, 1, 0]
correct_label = 0

passing = process_row(test_case_1)==(correct_row,correct_label)
print(passing)
if not passing:
    print("Expected")
    print((correct_row,correct_label))
    print("Got:")
    print(process_row(test_case_1))
