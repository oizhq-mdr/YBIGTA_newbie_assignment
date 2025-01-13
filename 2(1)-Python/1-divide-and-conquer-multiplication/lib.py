from __future__ import annotations
import copy


"""
TODO:
- __setitem__ 구현하기
- __pow__ 구현하기 (__matmul__을 활용해봅시다)
- __repr__ 구현하기
"""


class Matrix:
    MOD = 1000

    def __init__(self, matrix: list[list[int]]) -> None:
        self.matrix = matrix

    @staticmethod
    def full(n: int, shape: tuple[int, int]) -> Matrix: 
        return Matrix([[n] * shape[1] for _ in range(shape[0])])

    @staticmethod
    def zeros(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(0, shape)

    @staticmethod
    def ones(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(1, shape)

    @staticmethod
    def eye(n: int) -> Matrix:
        matrix = Matrix.zeros((n, n))
        for i in range(n):
            matrix[i, i] = 1
        return matrix

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.matrix), len(self.matrix[0]))

    def clone(self) -> Matrix:
        return Matrix(copy.deepcopy(self.matrix))

    def __getitem__(self, key: tuple[int, int]) -> int:
        return self.matrix[key[0]][key[1]]

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        # 구현하세요!
        """
        Args:
            - x (int): 행렬의 행 
            - y (int): 행렬의 열
        """
        x, y = key
        self.matrix[x][y] = value % self.MOD

    def __matmul__(self, matrix: Matrix) -> Matrix:
        x, m = self.shape
        m1, y = matrix.shape
        assert m == m1

        result = self.zeros((x, y))

        for i in range(x):
            for j in range(y):
                for k in range(m):
                    result[i, j] += self[i, k] * matrix[k, j]

        return result

    def __pow__(self, n: int) -> Matrix:
        """
        Args: 
            - n (int):

        Returns:
            - Matrix.eyes(x) (Matrix): 행렬의 0제곱인 경우 단위 행렬 반환
            - base (Matrix): 행렬의 1제곱인 경우 자기 자신의 복제본을 반환
                - 이때, 행렬의 원소가 1000인 케이스를 고려하여 행렬의 모든 원소에 대해 모듈러 연산
            - result (Matrix): 분할정복 알고리즘을 이용한 행렬의 제곱 연산
                - 거듭제곱 계산이 이루어질때마다 모듈러 연산을 통해 결과값 커지는 것 방지
        """
        x = self.shape[0]
        base = self.clone()
    
        if n == 0:
            return Matrix.eye(x)
        if n == 1:
            for i in range(x):
                    for j in range(x):
                        base[i, j] %= self.MOD
            return base

        result = Matrix.eye(x)

        while n > 0:
            if n % 2:
                result @= base
                for i in range(x):
                    for j in range(x):
                        result[i, j] %= self.MOD  # 모듈러 연산
            base @= base
            for i in range(x):
                for j in range(x):
                    base[i, j] %= self.MOD  # 모듈러 연산
            n //= 2
                
        return result

    def __repr__(self) -> str:
        """
        Return:
            - str: 각 행은 줄바꿈 문자('\n')로 구분되고, 행의 각 요소는 공백으로 구분된 문자열
        """
        result = "\n".join(" ".join(map(str, row)) for row in self.matrix)
        return result 