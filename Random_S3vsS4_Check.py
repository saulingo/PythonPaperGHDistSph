import math
import numpy as np
import random
import time

# Online Python - IDE, Editor, Compiler, Interpreter

SimplexList = [0, [[-1], [1]]]  # First we need a list of simplices.
for N in range(2, 200):
    verts = [SimplexList[N-1][i].copy() for i in range(len(SimplexList[N-1]))]
    for i in range(N):
        for j in range(N-1):
            verts[i][j] = verts[i][j]*math.sqrt(1-1/N**2)
        verts[i].append(-1/N)
    verts.append([0]*(N-1)+[1])
    SimplexList.append(verts)


# Let's project points of S^{n+1} to S^n
def pi(x):
    N = len(x)
    height = x[N-1]
    pi1 = x[0:N-1]
    return ([pi1[i]/((np.dot(pi1, pi1))**(1/2)) for i in range(len(pi1))]+[0])


# This is just the function f(x) from my document. 
def f(x):
    height=np.arccos(min(x,1))
    return(max(0,height+1-math.pi/2))


def SphDist(x, y):  # Spherical distances!
    # This is a fix because sometimes due to precision stuff the dot product is >1 and the program explodes.
    scprod = max(-1.0, min(1.0, np.dot(x, y)))
    return (np.arccos(scprod))


def afincomb(x, y, k):  # Returns spherical affine combinations kx+(1-k)y.
    alfa = np.arccos(np.dot(x, y))  # Just the angle between x and y
    xproj = np.array(x)-np.array(y)*np.dot(x, y)
    # vector perpendicular to y in direction of x.
    yperp = xproj/(np.dot(xproj, xproj)**(1/2))
    return (np.add(math.cos(k*alfa)*np.array(y), math.sin(k*alfa)*np.array(yperp)))
# Great! Now let's define our map, finally.


# Fn goes from S^{n+1} to S^n (F1 is sphere-->circumference), so we use a N+1-simplex
def Fn(n, x):
    x = np.array(x)/((np.dot(np.array(x), np.array(x)))**(1/2))
    verts = [SimplexList[n+1][i].copy() for i in range(len(SimplexList[n+1]))]
    for i in range(len(verts)):  # We need our vectors to be in R^{n+2}
        verts[i].append(0)
    imin = 0  # Let's find the closest point (with biggest scalar prod) to x.
    scmax = np.dot(x, verts[0])
    for i in range(1, n+2):
        scpi = np.dot(x, verts[i])
        if scmax < scpi:
            imin = i
            scmax = scpi
    p_i = verts[imin]
    return (afincomb(pi(x), p_i,f(x[-1])))


# Lista de puntos aleatorios en una esfera, cogiendo aleatoriamente altitud.
# Lista de puntos random en R^{n-1}xR de norma 1
def sample_spherical(ndim, npoints):
    vec = np.random.randn(npoints, ndim-1)
    # To experiment in high altitude, change this hvec by 0.99+0.01*np.random.rand(npoints, 1)
    hvec = np.random.rand(npoints, 1)
    for i in range(len(vec)):
        vec[i] = ((1-hvec[i]**2)**(1/2))*vec[i]/(np.dot(vec[i], vec[i])**(1/2))
    vec = np.hstack((vec, hvec))
    return vec


# Vectorized SphDist:
# Given two sets of N n-dim points, return an (N, N) matrix with the sphdist between each pair
def vecSphDist(ptsA, ptsB):
    # ptsX has shape (N, n)
    # Step 1: dot of all points against all points
    # dots[i][j] = dot product of pointA i & pointB j
    dots = ptsA @ ptsB.T
    # This is a fix because sometimes due to precision stuff the dot product is >1 and the program explodes.
    dots = np.clip(dots, -1, 1)
    return np.arccos(dots)


# Ahora empieza el programa de verdad. Vamos a hacer una función que experimente.


# n dimensión tal que nos situamos en R^{n+2}, N número de puntos aleatorios.
def experiment(n, N):
    # In this case it's not necessary to change the last coord of the points to be positive, by how I defined sample_spherical.
    Pts = sample_spherical(n+2, N)
    Pts[:, -1] = np.abs(Pts[:, -1])
    Imgs = np.array([Fn(n, Pts[i]) for i in range(len(Pts))])
    # Pts.shape = Imgs.shape == (N, n+2)
    print("done with preprocessing")

    max_distortion = 0
    # Chunk the computation into arrays of at most this size
    MAX_ARRAY_SIZE = 1000
    for i in range(0, N, MAX_ARRAY_SIZE):
        for j in range(i, N, MAX_ARRAY_SIZE):
            pts_dists = vecSphDist(Pts[i:i+MAX_ARRAY_SIZE], Pts[j:j+MAX_ARRAY_SIZE])
            imgs_dists = vecSphDist(Imgs[i:i+MAX_ARRAY_SIZE], Imgs[j:j+MAX_ARRAY_SIZE])
            distortions = np.abs(pts_dists - imgs_dists)
            max_distortion = max(max_distortion, np.max(distortions))
            #The following is mine, to check errors
            if np.max(distortions)>np.arccos(-1/(n+1)):
                for k in range(i,i+MAX_ARRAY_SIZE):
                    for l in range(j,j+MAX_ARRAY_SIZE):
                        minidistortion=abs(vecSphDist(Pts[k],Pts[l])-vecSphDist(Imgs[k],Imgs[l]))
                        if minidistortion>np.arccos(-1/(n+1)):
                            print("NOOOOOOOO")
                            print("\nPuntox:")
                            print(Pts[k])
                            print("\nImgx:")
                            print(Imgs[k])
                            print("\nPuntox':")
                            print(Pts[l])
                            print("\nImgx':")
                            print(Imgs[l])
                            print("\nSimplexVertices:")
                            print(SimplexList[n+1])

            #print(f"i, j: ({i}, {j})")

    # print(str(ptmax)+"\n")
    # print(str(imax)+"\n")
    print("Sample size: "+str(N)+" points from S^"+str(n+1))
    print("Distortion Lower bound: arccos(-1/" +
          str(n+1)+")="+str(np.arccos(-1/(n+1))))
    # The distortion should not be greater than arccos(-1/(n+1))
    print("Maximum distortion found:"+str(max_distortion)+"\n\n")




time0=time.time()
experiment(3,100000)
print("Time in seconds: "+str(time.time()-time0))