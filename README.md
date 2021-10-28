# Matrix

`Matrix` is a class written in Python that is built to handle Matrix multiplication and arithmatic operations.

## Functionality

- I have overloaded all the basic operations between two Matricies and even overloaded the insert and retrieval so that individual elements can be set or retrieved.
- There are 3 multiplication algorithms available to use, classic, divide and conquer, and Stassen.
- If you multiply two matrices using the `*` operand, it will default to classic multiplication
- I have included some static factory methods to easily generate Matrices of a specific value, zero, or random values.

## Performance Testing

- To test the performance, I have implemented code that will test the performance of each multiplication algorithm by averaging the performance 1,000 iterations of each size over set 20 sets.
- The application will print the performance of each algorithm and indicate the test it is currently performing.
