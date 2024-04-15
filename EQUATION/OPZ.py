def 

def OPZ(input):
    stack = []
    for i in range(len(input)):
        if input[i].isdigit():
            a = 0
            while input[i].isdigit():
                a = a*10+input[i]-'0'
                i=i+1
            stack.append(a)
        elif input[]