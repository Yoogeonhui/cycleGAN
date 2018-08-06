sizeInfo = input().split(' ')
w = int(sizeInfo[2])
h = int(sizeInfo[4])
backColorInfo = input().split(' ')
hex = int(backColorInfo[2],16)
NoF = int(input().split(' ')[3])
print(w)
print(h)
print(hex)
print(NoF)
mymap = {}
a = input().split(' ')
for i in range(1, 5):
    mymap[i] = a[i]

for i in range(1,NoF+1):
    printing={}
    inputString = input().split(' ')
    for j in range(1,5):
        printing[mymap[j]] = inputString[j]

    if printing['alpha'] == 'yes':
        printing['alpha'] = '1'
    else:
        printing['alpha'] = '0'
    print(printing['width']+'x'+printing['height']+', '+printing['duration']+', '+printing['alpha'])