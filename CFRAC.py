import math
import numpy as np
import time


def isqrt(n):
    '''Finder heltalsdelen af kvadratroden af n.'''
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def isprime(num):
    '''Tjekker hvorvidt et lille tal er et primtal.'''
    for n in range(2,int(num**0.5)+1):
        if num%n==0:
            return False
    return True


def primesunderbound(B):
    '''Returnerer en liste med alle primtal mindre end B.'''
    primes = []
    for i in range(2,B):
        if isprime(i):
            primes.append(i)
    return primes


def jacobi(n, p):
    '''Udregner Jacobi-symbolet (n/p) for p et ulige primtal.'''
    assert(p > 0 and p % 2 == 1)
    n = n % p
    t = 1
    while n != 0:
        while n % 2 == 0:
            n = n / 2
            r = p % 8
            if r == 3 or r == 5:
                t = -t
        n, p = p, n
        if n % 4 == 3 and p % 4 == 3:
            t = -t
        n = n % p
    if p == 1:
        return t
    else:
        return 0


def smallsquareresidues(kN,B):
    '''Returnerer en liste med -1, 2, og så alle primtal mindre end B,
       for hvilke kN er en kvadratisk rest.'''
    squares = [-1,2]
    for p in primesunderbound(B)[1:]:
        if jacobi(kN,p) != -1:
            squares.append(p)

    return squares


def bestkbelowb(N,b):
    '''Finder den værdi af k, for hvilken N er kvadratisk rest for
       flest primtal mindre 100, og hvor k er mindre end b.'''
    def goodres(k):
        count = 0
        for p in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97):
            if jacobi(k*N,p) == 1:
                count += 1
        return count
    bestk = 1
    bestcount = goodres(1)
    for k in range(2,b):
        if goodres(k) > bestcount:
            bestk, bestcount = k,goodres(k)
    return bestk


def findqs(k,N,steps):
    '''Genererer lister med værdier af blandt andet Q'erne og A'erne.
       Værdierne med negativt indeks genereres udenfor listerne,
       så det 0'te element i listen af Q'er Q_0.
       A_2 fortolkes som A med indeks -2.'''
    g = isqrt(k*N)
    A_2 = 0
    A_1 = 1
    Q_1 = k*N
    r_1 = g
    gplusP = [g]
    Q = [1]
    q = [gplusP[0] // Q[0]]
    r = [gplusP[0] % Q[0]]
    A = [(q[0]*A_1+A_2) % N]
    gplusP.append(2*g-r[0])
    Q.append(Q_1+q[0]*(r[0]-r_1))
    q.append(gplusP[1] // Q[1])
    r.append(gplusP[1] % Q[1])
    A.append((q[1]*A[0]+A_1) % N)
    gplusP.append(2*g-r[1])
    Q.append(Q[0]+q[1]*(r[1]-r[0]))

    for n in range(2,steps):
        q.append(gplusP[n] // Q[n])
        r.append(gplusP[n] % Q[n])
        A.append((q[n]*A[n-1]+A[n-2]) % N)
        gplusP.append(2*g-r[n])        
        Q.append(Q[n-1]+q[n]*(r[n]-r[n-1]))

    return gplusP,Q,q,r,A


def factorqs(base, Qs,N):
    '''Faktoriserer de fundne Q'er over en given primbase.
       Q'er der kan faktoriseres helt gemmes med pariteten
       af potenserne for hvert primtal.
       Early stopping anvendes.'''
    n = len(Qs)
    r = len(base)
    goodQs = []
    ns = []

    def tryprimes(a,b):
        nonlocal Q
        nonlocal alpha
        
        for p in base[a:b]:
            j = 0
            while Q % p == 0:
                j += 1
                Q = Q // p
            alpha.append(j % 2)
            if Q == 1:
                alpha += [0]*(r-len(alpha))
                ns.append(i)
                goodQs.append(alpha)
                break

    firstbound = isqrt(N) // 500
    secondbound = isqrt(N) // 20000000
        
    for i in range(1,n):
        Q = Qs[i]
        alpha = [i % 2]

        tryprimes(1,15)
        if 1 < Q < firstbound:
            tryprimes(15,95)
            if 1 < Q < secondbound:
                tryprimes(95,r)
        
    goodQs = np.array(goodQs)
        
    assert len(goodQs) == len(ns)
    
    return ns, goodQs


def reducematrix(Qmatrix):
    '''Foretager Gauss elimination på matricen af
       faktoriserede Q'er, og holder styr på,
       hvilke Q'er der svarer til en given 0-række.'''
    f = len(Qmatrix)
    r=len(Qmatrix[0])
    print('Dimensions of matrix:', f, 'x',r)

    historymatrix =np.identity(f,dtype = int)
    
    j = r-1
    
    while j >=0:
        pivot = -1
        for i,vect in enumerate(Qmatrix):
            if vect[j] == 1 and not any(vect[j+1:]):
                pivot = i
                break
        if pivot != -1:
            for m in range(pivot+1,f):
                if Qmatrix[m][j] == 1 and not any(Qmatrix[m][j+1:]):
                    Qmatrix[m] = Qmatrix[m]^Qmatrix[pivot]
                    historymatrix[m] = historymatrix[m]^historymatrix[pivot]
        j -= 1

    return Qmatrix, historymatrix


def checkcongruences(As,Qs,Qmatrix,historymatrix,ns,N):
    '''Hver 0-række i matricen af Q'er giver anledning til
       en kongruens. Hver af disse tjekkes for, om de
       producerer en ikke-triviel faktorisering af N'''
    f = len(historymatrix)
    sets = []
    for i in range(f):
        if not 1 in Qmatrix[i]:
            sets.append(i)

    print(len(sets), 'congruences found')
    for i in sets:
        qns = []
        for j in range(len(historymatrix[i])):
            if historymatrix[i][j]==1:
                qns.append(ns[j])
        Qproduct = 1
        Aproduct = 1
        for j in qns:
            Qproduct = (Qproduct * Qs[j])
            Aproduct = (Aproduct * As[j-1]) % N

        Qproduct = isqrt(Qproduct) % N

        potentialfactor = math.gcd(Aproduct-Qproduct, N)
        if potentialfactor != N and potentialfactor != 1:
            return 'Two factors are:', potentialfactor, N // potentialfactor

    return 'nothing found'


def CFRAC(k,N,n,bound):
    '''Kører samlet CFRAC-algoritmen med en given k-værdi,
       et givet N, et antal Q'er, der skal genereres, og
       en grænse på størrelsen af primtallene i basen.
       Tager tid på hver del af algoritmen undervejs.'''
    start = time.time()
    gplusP,Qs,q,r,As = findqs(k,N,n)
    end = time.time()

    print('Generating Qs:', round(end-start,1), 'seconds')

    start = time.time()
    goodprimes = smallsquareresidues(k*N,bound)
    for p in goodprimes[1:]:
        if N % p == 0:
            return 'Two factors are:', p, N//p
    ns, goodQsfactored = factorqs(goodprimes, Qs,N)
    
    if len(goodQsfactored)==0:
        return 'nothing found'
    end = time.time()

    print('Factoring Qs:', round(end-start,1), 'seconds')
    
    start = time.time()
    Qmatrix, historymatrix = reducematrix(goodQsfactored)
    end = time.time()

    print('Reducing matrix:', round(end-start,1), 'seconds')
    
    return checkcongruences(As,Qs,Qmatrix,historymatrix,ns,N) 


def CFRACauto(N):
    '''Kører CFRAC på et givet N, hvor k-værdien vælges smart,
       og antal Q'er og størrelsen på primtallene vælges først
       småt, og bliver så skaleret op, indtil en fakorisering findes.'''
    starttime = time.time()
    
    result = 'nothing found'
    k = bestkbelowb(N,1000)
    if N % k == 0:
        result = 'Two factors are:', k, N // k
    Qs = 2500
    primebound = 100
    while result == 'nothing found':
        print('No congruences have worked so far.')
        print('Trying k=', k, ', Number of Qs:',Qs, ', Biggest prime below:',primebound)
        result = CFRAC(k, N, Qs, primebound)
        Qs, primebound = Qs * 2, primebound * 2
    return *result, 'found in a total time of', round(time.time()-starttime,1), 'seconds'


F7 = 2**128 +1
F8 = 2**256 +1


start = time.time()
print(*CFRAC(157,F7,1200000,42000))
print('Took', round(time.time()-start,1), 'seconds overall.')


#print(*CFRACauto(F7))

#print(*CFRACauto(345446275827374619783746364523456757))
#print(*CFRAC(bestkbelowb(F8, 1000),F8,10000000,500000))




'''Nedenstående anvender de fundne kvotienter for
   kædebrøken for e til at estimere e. Allerede
   efter konvergent nummer 20 får man approksimationen
   410105312/150869313 der er korrekt til 16
   decimaler, trods nævneren i brøken kun er 9 cifre'''
def equotients(n):
    quotients = [2,1,2,]
    for i in range(2,n+1):
        quotients.append(1)
        quotients.append(1)
        quotients.append(i*2)
    return quotients[0:n+1]

def econvergents(n):
    qs = equotients(n)
    A = [qs[0]]
    B = [1]
    A.append(A[0]*qs[1]+1)
    B.append(B[0]*qs[1]+0)
    for q in qs[2:]:
        A.append(A[-1]*q+A[-2])
        B.append(B[-1]*q+B[-2])
    estimate = A[-1]/B[-1]
    convergents = zip(A,B)
    
    return estimate, *convergents
    
#print(*econvergents(20))


'''Nedenstående er en implementering af gentagen
   kvadrering og Pepins test, der anvender dette'''
def repeated_squaring(base, power, MOD):
    result = 1
    while power > 0:
        if power % 2 == 1:
            result = (result * base) % MOD

        power = power // 2
        base = (base * base) % MOD

    return result

def pepin(n):
    start = time.time()
        
    fn = 2**(2**n)+1
    power = 2**(2**n-1)
    remainder = repeated_squaring(3,power,fn)
    remainder = remainder - fn
    end = time.time()
    return 'prime!'*(remainder == -1)+'not prime...'*(remainder != -1)+' remainder is:', remainder, 'found in', round(end-start,3), 'seconds'

#print(*pepin(7))




