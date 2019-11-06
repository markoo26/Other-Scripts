
word_to_guess= 'Python'
letter = input('Make a guess of one letter: ')
show_chars = []
for character in range(0,len(word_to_guess)):
    if letter is in word_to_guess[character]:
        show_chars.append(character)