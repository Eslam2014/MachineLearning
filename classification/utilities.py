def multipy_matrix(matrix_1,matrix_2):
    '''
    multiply two matrices
    --------------------
    matrix_1 : 
        2 dimantions array 
    matrix_2 :
        2 dimantion array
    
    return :
        2 dimantion array matrix_1 rows * matrix2 columns
    '''
    
    if len(matrix_1[0]) != len(matrix_2): # check if columns of matrix_1 equal rows of matrix_2
        raise ValueError('Number of column of vector 1 must equals Number of rows in second vector')

    result_rows ,result_columns = len(matrix_1) , len(matrix_2[0])
    result_matrix = [[0]*result_columns for i in range(result_rows)]
    
    matrix_2_transpose = transpose_matrix(matrix_2)

    for row_num in range(result_rows):
        for column_num in range(result_columns):
            result_matrix[row_num][column_num] = dot_produnct(matrix_1[row_num],matrix_2_transpose[column_num])

    return result_matrix        
   

def transpose_matrix(matrix):
    '''
    transpose of a matrix is an operator which flips a matrix over its diagonal
    ---------------------------------------------------------------------------
    matrix : 
            disrable transpose matricx (multi dimantion array or vector)
        
    return :
            transposed matrix
    '''
    if  not type(matrix[0]) == list: # check if the matrix is vector
        result_rows = len(matrix)
        result_columns = 1
    else:
        result_rows = len(matrix[0])
        result_columns = len(matrix)

        
    transpose_matrix = [[0]*result_columns for i in range(result_rows)]
    for row in range(result_rows):
        transpose_matrix[row] = column(matrix,row)
            
    return transpose_matrix


def column(matrix, i):
    '''
    get  column from matrix 
    --------------------------------------
    matrix: 
            (multi dimantion array or vector)
    i :
        column index
        
    return :
        column list
    '''
    if  not type(matrix[0]) == list:
        return [matrix[i]]
    else:
        return [row[i] for row in matrix]

def dot_produnct(vector_1,vector_2):
    if len(vector_1) != len(vector_2):
        raise ValueError('vector_1 and vector_2 must have the same lenght')

    return sum( [vector_1[i]*vector_2[i] for i in range(len(vector_2))] )

def sum_two_vect(f_vect,s_vect):
    '''
    summation of two vectors
    -------------------------
    f_vect : 
            array of first vector
    s_vect :
            array of second vector
    return 
            summ vector
    '''

    return [x + y for x, y in zip(f_vect, s_vect)]

def subtract_two_vect(f_vect,s_vect):
    '''
    summation of two vectors
    -------------------------
    f_vect : 
            array of first vector
    s_vect :
            array of second vector
    return 
            summ vector
    '''

    return [x - y for x, y in zip(f_vect, s_vect)]

def sum_matrices(matrix_1,matrix_2):
    if len(matrix_1) != len(matrix_2) | len(matrix_1[0]) != len(matrix_2[0]) :
        raise ValueError("The two matrices must be the same size")

    result_matrix =  [[0]*len(matrix_1[0]) for i in range(len(matrix_1))]
    for row_num,m1_row in enumerate(matrix_1):
        result_matrix[row_num] = sum_two_vect(matrix_1[row_num],matrix_2[row_num])

    return result_matrix

def div_arr_by_num(arr,num):
    '''
    divide array by number
    ----------------------
    arr :
        array contains the numbers 
    num :
        number you want divide array elemets by it
    return :
        new array divided by the given number
    '''
    return [x / num for x in arr] 

def matmult(a,b):
    zip_b = zip(*b)
    # uncomment next line if python 3 : 
    zip_b = list(zip_b)
    return [[sum(ele_a*ele_b for ele_a, ele_b in zip(row_a, col_b)) 
             for col_b in zip_b] for row_a in a]       