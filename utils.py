
# God, please forgive me
def letter_to_number(letter):
    if letter == 'A':
        number = 0
    elif letter == 'B':
        number = 1
    elif letter == 'C':
        number = 2
    elif letter == 'D':
        number = 3
    elif letter == 'E':
        number = 4
    elif letter == 'F':
        number = 5
    elif letter == 'G':
        number = 6
    elif letter == 'H':
        number = 7
    elif letter == 'I':
        number = 8
    elif letter == 'J':
        number = 9
    elif letter == 'K':
        number = 10
    elif letter == 'L':
        number = 11
    elif letter == 'M':
        number = 12
    elif letter == 'N':
        number = 13
    elif letter == 'O':
        number = 14
    elif letter == 'P':
        number = 15
    elif letter == 'Q':
        number = 16
    elif letter == 'R':
        number = 17
    elif letter == 'S':
        number = 18
    elif letter == 'T':
        number = 19
    elif letter == 'U':
        number = 20
    elif letter == 'V':
        number = 21
    elif letter == 'W':
        number = 22
    elif letter == 'X':
        number = 23
    elif letter == 'Y':
        number = 24
    elif letter == 'Z':
        number = 25
    elif letter == 'del':
        number = 26
    elif letter == 'nothing':
        number = 27
    elif letter == 'space':
        number = 28
    else:
        number = -1  # Unknown
    return number


def number_to_letter(number):
    if number == 0:
        label = 'A'
    elif number == 1:
        label = 'B'
    elif number == 2:
        label = 'C'
    elif number == 3:
        label = 'D'
    elif number == 4:
        label = 'E'
    elif number == 5:
        label = 'F'
    elif number == 6:
        label = 'G'
    elif number == 7:
        label = 'H'
    elif number == 8:
        label = 'I'
    elif number == 9:
        label = 'J'
    elif number == 10:
        label = 'K'
    elif number == 11:
        label = 'L'
    elif number == 12:
        label = 'M'
    elif number == 13:
        label = 'N'
    elif number == 14:
        label = 'O'
    elif number == 15:
        label = 'P'
    elif number == 16:
        label = 'Q'
    elif number == 17:
        label = 'R'
    elif number == 18:
        label = 'S'
    elif number == 19:
        label = 'T'
    elif number == 20:
        label = 'U'
    elif number == 21:
        label = 'V'
    elif number == 22:
        label = 'W'
    elif number == 23:
        label = 'X'
    elif number == 24:
        label = 'Y'
    elif number == 25:
        label = 'Z'
    elif number == 26:
        label = 'del'
    elif number == 27:
        label = 'nothing'
    elif number == 28:
        label = 'space'
    else:
        label = 'unknown'
    return label