def priority(op):
    if op=='+' or op=='-':
        return 1
    elif op=='*' or op=='/':
        return 2
    else:
        return 0
    
def operation(a,b,c):
    if c=='+':
        return a+b
    elif c=='-':
        return b-a
    elif c=='*':
        return a*b
    else:
        return b/a

def OPZ(input):
    stack = [0]
    op = []
    i=0
    while i<len(input):
        if input[i].isdigit():
            a = 0
            while i<len(input) and input[i].isdigit():
                a = a*10+int(input[i])
                i=i+1
            
            stack.append(a)
            i-=1
        
        elif priority(input[i])>0:
            while len(op) > 0 and priority(input[i])<=priority(op[-1]):
                stack.append(operation(stack.pop(),stack.pop(),op.pop()))

            op.append(input[i])

        elif input[i]=='(':
            op.append('(')

        else:
            while op[-1]!='(':
                stack.append(operation(stack.pop(),stack.pop(),op.pop()))

            op.pop()
        i+=1
    while len(op) > 0:
        stack.append(operation(stack.pop(),stack.pop(),op.pop()))

    return stack[-1]
                
