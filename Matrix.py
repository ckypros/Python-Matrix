from __future__ import annotations
import time
import random


class Matrix:
    def __init__(self, array):
        self.array = array


    @staticmethod
    def ofZeroes(n: int, m: int=-1, ) -> Matrix:
        """Factory method generates a Matrix filled with 0's.

        If only 1 size parameter is passed, a square Matrix is generated.

        Parameters
        ----------
        n : int
            The # of cols required for the Matrix.

        m : int, optional
            The # of rows required for the Matrix.1
            If no m is assigned, it will will use value for n.

        Returns
        ------
        Matrix
            of size n by n, or m by n, filled with 0's
        """

        if m == -1: m = n # square matrix
        return Matrix.ofValue(m, n, 0)

    
    @staticmethod
    def ofRandoms(n: int, m: int=-1, minVal=0, maxVal=10000) -> Matrix:
        """Factory method generates a Matrix filled with random values.

        If only 1 size parameter is is passed, a square Matrix is generated.

        Parameters
        ----------
        n : int
            The # of cols required for the Matrix.

        m : int, optional
            The # of rows required for the Matrix.
            If no m is assigned, it will will use value for n.

        minVal : int
            The minimum random value expected in the Matrix
            If no minVal is supplied, it will default to 0

        maxVal : int
            The maximum random value expected in the Matrix
            If no maxVal is supplied, it will default to 10,000

        Returns
        ------
        Matrix
            of size n by n, or m by n, filled with random values
        """

        if m == -1: m = n # square matrix
        rndNum = lambda: random.randint(minVal, maxVal)
        return Matrix([[rndNum() for i in range(n)] for j in range(m)])


    @staticmethod
    def ofValue(n: int, m: int=-1, value=1) -> Matrix:
        """Factory method generates a Matrix filled with specified

        If only 1 size parameter is is passed, a square Matrix is generated.

        Parameters
        ----------
        n : int
            The # of cols required for the Matrix.

        m : int, optional
            The # of rows required for the Matrix.
            If no m is assigned, it will will use value for n.

        value : int/float
            Value to be assigned to all elements of the Matrix.
            If no value is supplied, it will default to 1

        Returns
        ------
        Matrix
            of size n by n, or m by n, filled with specified values
        """
        
        if m == -1: m = n # square matrix
        return Matrix([[value for i in range(n)] for j in range(m)])
    

    @staticmethod
    def stichQuadrants(a: Matrix, b: Matrix, c: Matrix, d: Matrix) -> Matrix:
        """Factory method stiches together 4 Matrices

        It is expected that 

        Parameters
        ----------
        a : Matrix
            that is expected to be stiched in NW quadrant
        
        b : Matrix
            that is expected to be stiched in NE quadrant

        c : Matrix
            that is expected to be stiched in SW quadrant

        d : Matrix
            that is expected to be stiched in SE quadrant

        Raises
        ------
        ValueError
            raised if the size of Matrices a, b, c, d not congruent


        Returns
        ------
        Matrix
            made from stiching together Matricies a, b, c, and d
        """

        if  not (a.getSize() == b.getSize() == c.getSize() == d.getSize()) :
            raise ValueError('Matricies a, b, c, d must be same size to be stiched together.')

        m = a.getM()
        array = []
        # Append rows of a and b together to form new Matrix
        for i in range(m):
            array.append(a[i] + b[i])
        # Append rows of c and d together to form new Matrix
        for i in range(m):
            array.append(c[i] + d[i])
        return Matrix(array)


    def __add__(self, other: Matrix) -> Matrix:
        """Method overloads add operation to add 2 Matricies

        Parameters
        ----------
        other : Matrix
            that is added with current matrix

        Raises
        ------
        ValueError
            raised if the size of Matrices self and other not the same

        Returns
        ------
        Matrix
            result of adding self + other Matrix
        """

        if self.getSize() != other.getSize() :
            raise ValueError('Matrix sizes do not match, can not add.')

        m, n = self.getSize()
        result = Matrix.ofZeroes(n, m)
        for i in range(m):
            for j in range(n):
                result[i, j] = self[i, j] + other[i, j]
        return result

    
    def __sub__(self, other: Matrix) -> Matrix:
        """Method overloads sub operation to subtract a Matrix

        Parameters
        ----------
        other : Matrix
            that is subtracted from current matrix

        Raises
        ------
        ValueError
            raised if the size of Matrices self and other not the same

        Returns
        ------
        Matrix
            result of subtracting other Matrix from self
        """

        if self.getSize() != other.getSize() :
            raise ValueError('Matrix sizes do not match, can not subtract.', self.getSize(), other.getSize())

        m, n = self.getSize()
        result = Matrix.ofZeroes(n, m)
        for i in range(n):
            for j in range(n):
                result[i, j] = self[i, j] - other[i, j]
        return result


    def __mul__(self, other: Matrix) -> Matrix:
        """Method overloads multiply operation to left multiply a Matrix

        Parameters
        ----------
        other : Matrix
            that is current matrix is left multiplied by

        Raises
        ------
        ValueError
            raised if the size of Matrices self and other not the same

        Returns
        ------
        Matrix
            result of left multiplying other Matrix by current Matrix
        """

        if self.getSize() != other.getSize() :
            raise ValueError('Matrix sizes do not match, can not multiply.')

        m = self.getM()
        p = other.getM()
        result = Matrix.ofZeroes(m, p)

        for i in range(m):
            for j in range(p):
                for k in range(m):
                    result[i, j] += self[i, k] * other[k, j]
        return result


    def __getitem__(self, key):
        """Method overloads array retrieval notation [] of Matrix

        Used to either an entire row or value of the Matrix

        Parameters
        ----------
        key : int or tuple
            If int, returns row number that is specified.
            If tuple in form [i, j], return array value at [i, j]

        Raises
        ------
        ValueError
            raised if key is not of of the appropriate type
        IndexError
            raised if invalid row index is specified in key
        IndexErorr
            raised if invalid element location is specified in key

        Returns
        ------
        Row or Number
            row / element retrieved using the specified key
        """
        
        if type(key) == int:
            try:
                return self.array[key]
            except:
                raise IndexError("Row specified not except")
        elif type(key) == tuple and len(key) == 2:
            try:
                i, j = key
                return self.array[i][j]
            except:
                raise IndexError("Specified Matrix element location does not exist")
        else:
            raise ValueError("Key parameter must be int or tuple in form of (i, j).")


    def __setitem__(self, key: tuple[int, int], new_value) -> None:
        """Method overloads array set notation [] of Matrix

        Used to set the value of individual element of a Matrix

        Parameters
        ----------
        key : tuple(int, int)
            First int of tuple specifies row index
            Second int of tuple specifies col index

        """

        i, j = key
        self.array[i][j] = new_value


    def __repr__(self) -> str:
        """Method overloads string representation of Matrix

            Prints Matrix, row by row on each line.
        """

        return '\n'.join([''.join(['{:7}'.format(item) for item in row]) for row in self.array])


    def getSize(self) -> tuple[int, int]:
        """Method retreives row and col dimensions of the Matrix

        Return
        ----------
        tuple(int, int)
            First int of tuple specifies # of rows
            Second int of tuple specifies # of cols

        """

        return (self.getM(), self.getN())


    def getM(self) -> int:
        """Method retreives row dimension of the Matrix

        Return
        ----------
        int
            the number of rows in the Matrix

        """

        return len(self.array)
    

    def getN(self) -> int:
        """Method retreives column dimension of the Matrix

        Return
        ----------
        int
            the number of columns in the Matrix

        """

        return len(self.array[0])

    
    def splitMatrixIntoQuadrants(self) -> tuple[Matrix, Matrix, Matrix, Matrix]:
        """Method splits Matrix into 4 equally sized quandrants

        Return
        ----------
        tuple[Matrix, Matrix, Matrix, Matrix]
            contains 4 Matrices in order of NW, NE, SW, SE

        """

        n = self.getN()
        n2 = n // 2
        a = self.getSubMatrix([0,n2], [0,n2])
        b = self.getSubMatrix([0,n2], [n2,n])
        c = self.getSubMatrix([n2, n], [0,n2])
        d = self.getSubMatrix([n2,n], [n2,n])
        return a, b, c, d


    def getSubMatrix(self, rows: tuple[int, int], cols: tuple[int, int]) -> Matrix:
        """Method gets sub-Matrix from specified rows and cols

        End row and end col is not inclusive.

        Parameters
        ----------
        rows : tuple(int, int)
            First int of tuple specifies starting row index
            Second int of tuple specifies ending row index

        cols : tuple(int, int)
            First int of tuple specifies starting column index
            Second int of tuple specifies ending column index

        Return
        ----------
        Matrix
            derived of elements contained within specified rows/cols 
            of current Matrix

        """

        startRow, endRow = rows
        startCol, endCol = cols
        
        # Grab elements row by row, col by col to build new Matrix
        array = []
        for i in range(startRow, endRow):
            row = []
            for j in range(startCol, endCol):
                row.append(self[i, j])
            array.append(row)
        return Matrix(array)
    

    def divideAndConquerMultiply(self, other: Matrix) -> Matrix:
        """Method left multiplies current matrix by other

        Implements divide and conquer algorithm.

        Parameters
        ----------
        other : Matrix
            Matrix to left multiply 

        Return
        ----------
        Matrix
            result from Matrix multiplication

        """

        # Split each Matrix into 4 quadrants 
        a, b, c, d = self.splitMatrixIntoQuadrants()
        e, f, g, h = other.splitMatrixIntoQuadrants()

        # Generate resulting Matrix quadrants
        c11 = (a * e) + (b * g)
        c12 = (a * f) + (b * h)
        c21 = (c * e) + (d * g)
        c22 = (c * f) + (d * h)

        return self.stichQuadrants(c11, c12, c21, c22)


    def strassenMultiply(self, other: Matrix) -> Matrix:
        """Method left multiplies current matrix by other

        Implements Strassen algorithm using recursion.

        Parameters
        ----------
        other : Matrix
            Matrix to left multiply 

        Return
        ----------
        Matrix
            result from Matrix multiplication

        """

        # If size of Matrix is 1, then just multiply them together
        if self.getM() == 1:
            return self * other

        # Split each Matrix into 4 quadrants 
        a, b, c, d = self.splitMatrixIntoQuadrants()
        e, f, g, h = other.splitMatrixIntoQuadrants()

        # Recursively split Matrix up until it is of size 1
        # Then perform execute algorithm to reassemble resulting Matrix
        p1 = a.strassenMultiply(f - h)
        p2 = (a + b).strassenMultiply(h)
        p3 = (c + d).strassenMultiply(e)
        p4 = d.strassenMultiply(g - e)
        p5 = (a + d).strassenMultiply(e + h)
        p6 = (b - d).strassenMultiply(g + h)
        p7 = (a - c).strassenMultiply(e + f)

        c11 = p5 + p4 - p2 + p6
        c12 = p1 + p2
        c21 = p3 + p4
        c22 = p1 + p5 - p3 - p7

        return self.stichQuadrants(c11, c12, c21, c22)


# Example usages

# Create 2 Matrices of size 8 x 8, all Random #'s from 0-100
m1 = Matrix.ofRandoms(8, maxVal=100) 
m2 = Matrix.ofRandoms(8, maxVal=100) 
m3 = m1 + m2
m4 = m1 - m2
m5 = m1 * m2

# Print each Matrix
print("m1", m1, "m2", m2, "m3", m3, "m4", m4, "m5", m5, sep='\n')

# Test other matrix multiplication algorithms.  m5 should be same as m6 and m7
m6 = m1.divideAndConquerMultiply(m2)
m7 = m1.strassenMultiply(m2)
print("m6", m6, "m7", m7, sep="\n")


# Test efficiency of algorithm
iterations = 1000
sets = 20
for n in [2, 4, 8, 16, 32, 64, 128]:
    print(f"\nFOR n={n}:")
    start = time.time()
    for i in range(iterations):
        for set in range(sets):
            # random.seed(i)
            m1 = Matrix.ofRandoms(n, maxVal=100)
            m2 = Matrix.ofRandoms(n, maxVal=100)
            m1 * m2
    end = time.time()
    print("For regular multiplication:")
    print((end - start) / sets)

    start = time.time()
    for i in range(iterations):
        for set in range(sets):
        # random.seed(i)
            m1 = Matrix.ofRandoms(n, maxVal=100)
            m2 = Matrix.ofRandoms(n, maxVal=100)
            m1.divideAndConquerMultiply(m2)
    end = time.time()
    print("For divide and conquer multiplication:")
    print((end - start) / sets)

    start = time.time()
    for i in range(iterations):
        for set in range(sets):
        # random.seed(i)
            m1 = Matrix.ofRandoms(n, maxVal=100)
            m2 = Matrix.ofRandoms(n, maxVal=100)
            m1.strassenMultiply(m2)
    end = time.time()
    print("For Strassen multiplication:")
    print((end - start) / sets)
