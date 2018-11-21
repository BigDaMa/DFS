#one iteration

#assumptions
# max higher order function is binary (see GroupBy (1 Key) Then Apply Function on 1 grouped feature)

U = 30 # vs 5 #number of unary transformations
Fi = 12 # heart # number attributes in the dataset
B = 5# number of binary transformations
B += 5 # let's assume group by is binary


Fui = Fi * U #unary

cand = Fui + Fi
Foi = cand * (cand-1) * B #binary

Foui = Foi * U #unary (binary)

all = Fui + Foi + Foui



print all