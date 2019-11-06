import random

number = str(random.randrange(1000,9999,1))

attempts=user_value=cows=bulls= 0

while cows != 4:
    cows = bulls = 0
    user_value = input('Make a guess: ')
    user_value = str(user_value)
    for character in range(0,len(number)):
        if number[character] == user_value[character]:
            cows +=1
        else:
            bulls +=1
    attempts += 1    

    print ('You have got ' + str(cows) + ' cows and ' + str(bulls) + ' bulls')
    print ('This was your ' +  str(attempts) + ' attempt')

print('You guessed the number, congrats! To do that you needed ' +
      str(attempts) + ' attempts')