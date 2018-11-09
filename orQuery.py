# Bismillahir Rahmanir Rahim
# Rabbi Zidni Ilma


import numpy as np
import matplotlib.pyplot as plt

step = 0

class Node(object):

    def __init__(self, localValue=0, globalValue=0, localLazy=0, globalLazy=0):
        self.localValue = localValue
        self.globalValue = globalValue
        self.localLazy = localLazy
        self.globalLazy = globalLazy


class SegmentTree2D(object):

    def __init__(self, n, m):

        self.n = n
        self.m = m
        self.tree = []

        for i in range(4 * n):

            li = []

            for j in range(4 * m):
                li.append(Node())

            self.tree.append(li)

    def update(self, qxLo, qxHi, qyLo, qyHi, v):

        self.updateByX((1, 1), 1, self.n, qxLo, qxHi, qyLo, qyHi, v)

    def query(self, qxLo, qxHi, qyLo, qyHi):

        return self.queryByX((1, 1), 1, self.n, qxLo, qxHi, qyLo, qyHi)

    def updateByX(self, nodeID, xLo, xHi, qxLo, qxHi, qyLo, qyHi, v):

        global step
        step += 1

        if (qxLo <= xLo and xHi <= qxHi):  # x is covered

            self.updateByY((nodeID[0], 1), xLo, xHi, 1, self.m, qxLo, qxHi, qyLo, qyHi, v, True)
            return

        elif (xLo > qxHi or xHi < qxLo):

            return

        else:

            xMid = (xLo + xHi) // 2
            left = (nodeID[0] * 2, 1)
            right = (left[0] + 1, 1)

            self.updateByX(left, xLo, xMid, qxLo, qxHi, qyLo, qyHi, v)
            self.updateByX(right, xMid + 1, xHi, qxLo, qxHi, qyLo, qyHi, v)

            txLo = max(qxLo, xLo)
            txHi = min(qxHi, xHi)

            self.updateByY((nodeID[0], 1), xLo, xHi, 1, self.m, qxLo, qxHi, qyLo, qyHi,
                           v , False)

    def updateByY(self, nodeID, xLo, xHi, yLo, yHi, qxLo, qxHi, qyLo, qyHi, v, covered):

        # covered = 1 is x part is contained within the query segment
        # v is already scaled from the previous layer

        global step
        step += 1

        if (qyLo <= yLo and yHi <= qyHi):  # y is covered

            if (covered):

                self.tree[nodeID[0]][nodeID[1]].globalLazy |= v
                self.tree[nodeID[0]][nodeID[1]].globalValue |= v 

            else:

                self.tree[nodeID[0]][nodeID[1]].localLazy |= v
                self.tree[nodeID[0]][nodeID[1]].localValue |= v 

            return


        elif (yHi < qyLo or qyHi < yLo):

            return


        else:

            yMid = (yLo + yHi) // 2
            left = (nodeID[0], nodeID[1] * 2)
            right = (left[0], left[1] + 1)

            self.updateByY(left, xLo, xHi, yLo, yMid, qxLo, qxHi, qyLo, qyHi, v, covered)
            self.updateByY(right, xLo, xHi, yMid + 1, yHi, qxLo, qxHi, qyLo, qyHi, v, covered)

            self.tree[nodeID[0]][nodeID[1]].localValue = self.tree[left[0]][left[1]].localValue | self.tree[right[0]][right[1]].localValue | (self.tree[nodeID[0]][nodeID[1]].localLazy) 
            self.tree[nodeID[0]][nodeID[1]].globalValue = self.tree[left[0]][left[1]].globalValue | self.tree[right[0]][right[1]].globalValue | (self.tree[nodeID[0]][nodeID[1]].globalLazy) 

    def queryByX(self, nodeID, xLo, xHi, qxLo, qxHi, qyLo, qyHi):

        global step
        step += 1

        if (qxLo <= xLo and xHi <= qxHi):  # y is covered

            return self.queryByY((nodeID[0], 1), xLo, xHi, 1, self.m, qxLo, qxHi, qyLo, qyHi, 0)

        elif (xLo > qxHi or xHi < qxLo):

            return 0

        else:

            xMid = (xLo + xHi) // 2
            left = (nodeID[0] * 2, nodeID[1])
            right = (left[0] + 1, left[1])

            txLo = max(xLo, xLo) # not necessary
            txHi = min(xHi, xHi) # not necessary
            #print( xLo, xHi, self.queryByY(nodeID,xLo,xHi,1, self.m, qxLo, qxHi, qyLo, qyHi, 0))
            return self.queryByY(nodeID,xLo,xHi,1, self.m, qxLo, qxHi, qyLo, qyHi, 0) | self.queryByX(left, xLo, xMid, qxLo, qxHi, qyLo, qyHi) | self.queryByX(right, xMid+1, xHi, qxLo, qxHi, qyLo, qyHi)
            #return self.queryByY(left, xLo, xMid, 1, self.m, qxLo, qxHi, qyLo, qyHi, 0) + self.queryByY(right, xMid + 1,xHi, 1, self.m,qxLo, qxHi,qyLo, qyHi, 0)

    def queryByY(self, nodeID, xLo, xHi, yLo, yHi, qxLo, qxHi, qyLo, qyHi, lazy):

        global step
        step += 1

        if (qyLo <= yLo and yHi <= qyHi):  # y is covered

            if (qxLo <= xLo and xHi <= qxHi):  # x is covered

                return self.tree[nodeID[0]][nodeID[1]].localValue | self.tree[nodeID[0]][nodeID[1]].globalValue | lazy 

            elif (xLo > qxHi or xHi < qxLo):  # This will never happen

                return 0

            else:

   
                return (self.tree[nodeID[0]][nodeID[1]].globalValue | lazy )


        elif (yLo > qyHi or yHi < qyLo):

            return 0

        else:

            yMid = (yLo + yHi) // 2
            left = (nodeID[0], nodeID[1] * 2)
            right = (left[0], left[1] + 1)

            if (qxLo <= xLo and xHi <= qxHi):  # x is covered

                return self.queryByY(left, xLo, xHi, yLo, yMid, qxLo, qxHi, qyLo, qyHi,lazy | self.tree[nodeID[0]][nodeID[1]].globalLazy | self.tree[nodeID[0]][nodeID[1]].localLazy) | self.queryByY(right, xLo, xHi, yMid + 1, yHi, qxLo,qxHi, qyLo, qyHi,lazy | self.tree[nodeID[0]][nodeID[1]].globalLazy | self.tree[nodeID[0]][nodeID[1]].localLazy)

            # return self.tree[nodeID[0]][nodeID[1]].localValue + self.tree[nodeID[0]][nodeID[1]].globalValue + lazy * (xHi - xLo + 1) * (yHi - yLo + 1)

            elif (xLo > qxHi or xHi < qxLo):  # This will never happen

                return 0

            else:

                return self.queryByY(left, xLo, xHi, yLo, yMid, qxLo, qxHi, qyLo, qyHi,lazy | self.tree[nodeID[0]][nodeID[1]].globalLazy) | self.queryByY(right, xLo, xHi,yMid + 1, yHi,qxLo, qxHi,qyLo, qyHi,lazy | self.tree[nodeID[0]][nodeID[1]].globalLazy)


class BruteForce(object):

    def __init__(self, n, m):

        self.grid = []

        for i in range(n + 1):

            ti = []

            for j in range(m + 1):
                ti.append(0)

            self.grid.append(ti)

    def update(self, qxLo, qxHi, qyLo, qyHi, v):

        for i in range(qxLo, qxHi + 1):

            for j in range(qyLo, qyHi + 1):
                self.grid[i][j] |= v

    def query(self, qxLo, qxHi, qyLo, qyHi):

        ans = 0

        for i in range(qxLo, qxHi + 1):

            for j in range(qyLo, qyHi + 1):

                ans |= self.grid[i][j]

        return ans

def timingSimulation():

	N = 5

	global step

	li = []

	segTree = SegmentTree2D(N,N)

	for N in range(5,500):
		
		step = 0	

		print(N)

		segTree = SegmentTree2D(N,N)

		qxLo = np.random.randint(1, N)
		qxHi = np.random.randint(1, N)
		qyLo = np.random.randint(1, N)
		qyHi = np.random.randint(1, N)
		v = np.random.randint(1, N)

		if (qxLo > qxHi):
			(qxHi, qxLo) = (qxLo, qxHi)

		if (qyLo > qyHi):
			(qyHi, qyLo) = (qyLo, qyHi)


		segTree.update(qxLo, qxHi, qyLo, qyHi, v)


		#print(('update', qxLo, qxHi, qyLo, qyHi, v))

		for i in range(100):

			qxLo = np.random.randint(1, N)
			qxHi = np.random.randint(1, N)
			qyLo = np.random.randint(1, N)
			qyHi = np.random.randint(1, N)

			if (qxLo > qxHi):
				(qxHi, qxLo) = (qxLo, qxHi)

			if (qyLo > qyHi):
				(qyHi, qyLo) = (qyLo, qyHi)


			#print(('query', qxLo, qxHi, qyLo, qyHi))


			segTree.query(qxLo, qxHi, qyLo, qyHi)
		li.append(step/100)

	plt.plot(li)
	plt.show()

def simulation():
    N = 200

    segTree = SegmentTree2D(N, N)
    rectGrid = BruteForce(N, N)

    while (True):

        print('.')

        qxLo = np.random.randint(1, N)
        qxHi = np.random.randint(1, N)
        qyLo = np.random.randint(1, N)
        qyHi = np.random.randint(1, N)
        v = np.random.randint(1, 255)

        if (qxLo > qxHi):
            (qxHi, qxLo) = (qxLo, qxHi)

        if (qyLo > qyHi):
            (qyHi, qyLo) = (qyLo, qyHi)

        segTree.update(qxLo, qxHi, qyLo, qyHi, v)
        rectGrid.update(qxLo, qxHi, qyLo, qyHi, v)

        print(('update', qxLo, qxHi, qyLo, qyHi, v))

        for i in range(20):

            qxLo = np.random.randint(1, N)
            qxHi = np.random.randint(1, N)
            qyLo = np.random.randint(1, N)
            qyHi = np.random.randint(1, N)

            if (qxLo > qxHi):
                (qxHi, qxLo) = (qxLo, qxHi)

            if (qyLo > qyHi):
                (qyHi, qyLo) = (qyLo, qyHi)

            #print(('query', qxLo, qxHi, qyLo, qyHi))

            if (abs(segTree.query(qxLo, qxHi, qyLo, qyHi) - rectGrid.query(qxLo, qxHi, qyLo, qyHi)) <= 1e-3):
                pass

            else:
                print(('query', qxLo, qxHi, qyLo, qyHi))
                print(segTree.query(qxLo, qxHi, qyLo, qyHi), rectGrid.query(qxLo, qxHi, qyLo, qyHi))

                print('ERR')
                t = input('')


if __name__ == '__main__':

    np.random.seed(3)

    simulation()

    bf = BruteForce(3,3)

    bf.update(1,2,1,2,5)

    print(bf.query(1,2,1,2))
    print(bf.query(1,3,1,3))

    bf.update(3,3,3,3,15)

    print(bf.query(1,2,1,2))
    print(bf.query(1,3,1,3))

    #timingSimulation()

    input('')

    rectGrid = BruteForce(6,6)
    rectGrid.update(1, 3, 2, 4, 1)
    rectGrid.update(1, 1, 1, 2, 2)
    rectGrid.update(2, 4, 2, 5, 1)
    rectGrid.update(4, 5, 2, 3, 2)
    rectGrid.update(1, 3, 1, 3, 3)

    for i in rectGrid.grid:
        print(i)

    simulation()

    segTree = SegmentTree2D(6, 6)
    segTree.update(1, 3, 2, 4, 1)
    segTree.update(1, 1, 1, 2, 2)
    segTree.update(2, 4, 2, 5, 1)
    segTree.update(4, 5, 2, 3, 2)
    segTree.update(1, 3, 1, 3, 3)
    print(segTree.query(3,6,1,1))

    while True:
        qxLo = int(input(''))
        qxHi = int(input(''))
        qyLo = int(input(''))
        qyHi = int(input(''))

        print(segTree.query(qxLo, qxHi, qyLo, qyHi))



    segTree.update(1, 3, 2, 4, 1)
    #segTree.update(1, 1, 1, 2, 2)
    #segTree.update(2, 4, 2, 5, 1)
    print(segTree.query(1,3,1,3))
    print(segTree.query(1, 3, 2, 3))
    print(segTree.query(2, 4, 2, 3))
    print(segTree.query(1, 5, 3, 5))

    segTree.update(1, 5, 1, 5, 4)

    segTree.update(7, 8, 2, 6, 8)

    segTree.update(2, 4, 3, 4, 2)

    segTree.update(2, 3, 1, 8, 2)

    print(segTree.query(1, 5, 1, 6))

    while True:
        qxLo = int(input(''))
        qxHi = int(input(''))
        qyLo = int(input(''))
        qyHi = int(input(''))

        print(segTree.query(qxLo, qxHi, qyLo, qyHi))






