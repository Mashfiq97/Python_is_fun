
import random 


random_num = random.randint(1,101)
print(random_num)

count = 0


guess = -1

while(guess != random_num):
    
    
    guess_num = int(float(input("Please enter the guess between 1 and 100:")))
    
    if guess_num < random_num:
        print("Please higher your Guessing")
    
    elif guess_num > random_num:
        print("Please lower your Guessing")
    
    elif guess_num == random_num:
        print("The number is:" , random_num)
        print("Horray,You have guessed correctly!!!!!")
        count = count + 1
       
        break
    
    count = count + 1
    
print("You have guessed the word in the {} steps".format(count))

