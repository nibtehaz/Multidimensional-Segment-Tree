"""
	بسم الله الرحمن الرحيم
	ربِّ زِدْنِي عِلْماً 

	Author : Nabil Ibtehaz

	This is our implementation of the proposed 2D Segment Tree
	in solving range multiplication query
"""


import numpy as np

step = 0


class Node(object):
	def __init__(self, localValue=1.0, globalValue=1.0, localLazy=1.0, globalLazy=1.0):
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
						   v ** ( ((1.0 * (txHi - txLo + 1)) / (xHi - xLo + 1)) ), False)

	def updateByY(self, nodeID, xLo, xHi, yLo, yHi, qxLo, qxHi, qyLo, qyHi, v, covered):

		# covered = 1 is x part is contained within the query segment
		# v is already scaled from the previous layer

		global step
		step += 1

		if (qyLo <= yLo and yHi <= qyHi):  # y is covered

			if (covered):

				self.tree[nodeID[0]][nodeID[1]].globalLazy *= v
				self.tree[nodeID[0]][nodeID[1]].globalValue *= v ** ((xHi - xLo + 1) * (yHi - yLo + 1))

			else:

				self.tree[nodeID[0]][nodeID[1]].localLazy *= v
				self.tree[nodeID[0]][nodeID[1]].localValue *= v ** ((xHi - xLo + 1) * (yHi - yLo + 1))

			return


		elif (yHi < qyLo or qyHi < yLo):

			return


		else:

			yMid = (yLo + yHi) // 2
			left = (nodeID[0], nodeID[1] * 2)
			right = (left[0], left[1] + 1)

			self.updateByY(left, xLo, xHi, yLo, yMid, qxLo, qxHi, qyLo, qyHi, v, covered)
			self.updateByY(right, xLo, xHi, yMid + 1, yHi, qxLo, qxHi, qyLo, qyHi, v, covered)

			self.tree[nodeID[0]][nodeID[1]].localValue = self.tree[left[0]][left[1]].localValue * self.tree[right[0]][right[1]].localValue * (self.tree[nodeID[0]][nodeID[1]].localLazy) ** ((xHi - xLo + 1) * (yHi - yLo + 1))
			self.tree[nodeID[0]][nodeID[1]].globalValue = self.tree[left[0]][left[1]].globalValue * self.tree[right[0]][right[1]].globalValue * (self.tree[nodeID[0]][nodeID[1]].globalLazy) ** ((xHi - xLo + 1) * (yHi - yLo + 1))

	def queryByX(self, nodeID, xLo, xHi, qxLo, qxHi, qyLo, qyHi):

		global step
		step += 1

		if (qxLo <= xLo and xHi <= qxHi):  # x is covered

			return self.queryByY((nodeID[0], 1), xLo, xHi, 1, self.m, qxLo, qxHi, qyLo, qyHi, 1.0)

		elif (xLo > qxHi or xHi < qxLo):

			return 1.0

		else:

			xMid = (xLo + xHi) // 2
			left = (nodeID[0] * 2, nodeID[1])
			right = (left[0] + 1, left[1])

			txLo = max(xLo, xLo)  # not necessary
			txHi = min(xHi, xHi)  # not necessary
			# print( xLo, xHi, self.queryByY(nodeID,xLo,xHi,1, self.m, qxLo, qxHi, qyLo, qyHi, 0))
			return self.queryByY(nodeID, xLo, xHi, 1, self.m, qxLo, qxHi, qyLo, qyHi, 1.0) ** (((1.0 * (txHi - txLo + 1)) / (xHi - xLo + 1))) * self.queryByX(left, xLo, xMid, qxLo, qxHi, qyLo, qyHi) * self.queryByX(right, xMid + 1, xHi, qxLo, qxHi, qyLo, qyHi)
		# return self.queryByY(left, xLo, xMid, 1, self.m, qxLo, qxHi, qyLo, qyHi, 0) + self.queryByY(right, xMid + 1,xHi, 1, self.m,qxLo, qxHi,qyLo, qyHi, 0)

	def queryByY(self, nodeID, xLo, xHi, yLo, yHi, qxLo, qxHi, qyLo, qyHi, lazy):

		global step
		step += 1

		if (qyLo <= yLo and yHi <= qyHi):  # y is covered

			if (qxLo <= xLo and xHi <= qxHi):  # x is covered

				return self.tree[nodeID[0]][nodeID[1]].localValue * self.tree[nodeID[0]][nodeID[1]].globalValue * lazy ** ( (xHi - xLo + 1) * (yHi - yLo + 1) )

			elif (xLo > qxHi or xHi < qxLo):  # This will never happen

				return 1.0

			else:

				txLo = max(qxLo, xLo)
				txHi = min(qxHi, xHi)

				return (self.tree[nodeID[0]][nodeID[1]].globalValue ** (  (txHi - txLo + 1) / (xHi - xLo + 1) ) ) * lazy ** ((txHi - txLo + 1) * (yHi - yLo + 1))


		elif (yLo > qyHi or yHi < qyLo):

			return 1.0

		else:

			yMid = (yLo + yHi) // 2
			left = (nodeID[0], nodeID[1] * 2)
			right = (left[0], left[1] + 1)

			if (qxLo <= xLo and xHi <= qxHi):  # x is covered

				return self.queryByY(left, xLo, xHi, yLo, yMid, qxLo, qxHi, qyLo, qyHi, lazy * self.tree[nodeID[0]][nodeID[1]].globalLazy * self.tree[nodeID[0]][nodeID[1]].localLazy) * self.queryByY(right, xLo, xHi, yMid + 1, yHi, qxLo, qxHi, qyLo, qyHi,
																																																	   lazy * self.tree[nodeID[0]][nodeID[1]].globalLazy * self.tree[nodeID[0]][nodeID[1]].localLazy)

			# return self.tree[nodeID[0]][nodeID[1]].localValue + self.tree[nodeID[0]][nodeID[1]].globalValue + lazy * (xHi - xLo + 1) * (yHi - yLo + 1)

			elif (xLo > qxHi or xHi < qxLo):  # This will never happen

				return 1.0

			else:

				return self.queryByY(left, xLo, xHi, yLo, yMid, qxLo, qxHi, qyLo, qyHi, lazy * self.tree[nodeID[0]][nodeID[1]].globalLazy) * self.queryByY(right, xLo, xHi, yMid + 1, yHi, qxLo, qxHi, qyLo, qyHi, lazy * self.tree[nodeID[0]][nodeID[1]].globalLazy)


class BruteForce(object):
	def __init__(self, n, m):

		self.grid = []

		for i in range(n + 1):

			ti = []

			for j in range(m + 1):
				ti.append(1.0)

			self.grid.append(ti)

	def update(self, qxLo, qxHi, qyLo, qyHi, v):

		for i in range(qxLo, qxHi + 1):

			for j in range(qyLo, qyHi + 1):
				self.grid[i][j] *= v

	def query(self, qxLo, qxHi, qyLo, qyHi):

		ans = 1.0

		for i in range(qxLo, qxHi + 1):

			for j in range(qyLo, qyHi + 1):
				ans *= self.grid[i][j]

		return ans




def simulation():
	N = 200

	segTree = SegmentTree2D(N, N)
	rectGrid = BruteForce(N, N)

	while (True):		

		qxLo = np.random.randint(1, N)
		qxHi = np.random.randint(1, N)
		qyLo = np.random.randint(1, N)
		qyHi = np.random.randint(1, N)
		v = np.random.randint(40,60) /100

		if (qxLo > qxHi):
			(qxHi, qxLo) = (qxLo, qxHi)

		if (qyLo > qyHi):
			(qyHi, qyLo) = (qyLo, qyHi)

		segTree.update(qxLo, qxHi, qyLo, qyHi, v)
		rectGrid.update(qxLo, qxHi, qyLo, qyHi, v)

		print('-'*40)
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

			if (abs(segTree.query(qxLo, qxHi, qyLo, qyHi) - rectGrid.query(qxLo, qxHi, qyLo, qyHi)) <= 1e-5):
				print(('query', qxLo, qxHi, qyLo, qyHi))
				print('Tree : '+str(segTree.query(qxLo, qxHi, qyLo, qyHi)) + ' , Brute Force : ' +str(rectGrid.query(qxLo, qxHi, qyLo, qyHi)) + ' , Difference : ' + str(abs(segTree.query(qxLo, qxHi, qyLo, qyHi) - rectGrid.query(qxLo, qxHi, qyLo, qyHi))))
				
			else:
				print(('query', qxLo, qxHi, qyLo, qyHi))
				print('Tree : '+str(segTree.query(qxLo, qxHi, qyLo, qyHi)) + ' , Brute Force : ' +str(rectGrid.query(qxLo, qxHi, qyLo, qyHi)) + ' , Difference : ' + str(abs(segTree.query(qxLo, qxHi, qyLo, qyHi) - rectGrid.query(qxLo, qxHi, qyLo, qyHi))))

				raise Exception('ERROR')



if __name__ == '__main__':

	np.random.seed(3)

	simulation()

	