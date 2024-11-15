#####################################################################
#          ,
#      /\^/`\
#     | \/   |
#     | |    |                 CPA ON ROMULUS 
#     \ \    /                                                _ _
#      '\\//'                                               _{ ' }_
#        ||                  Deepak S & Monish             { `.!.` }
#        ||              <deepaksridhar13@gmail.com>       ',_/Y\_,'
#        ||  ,                                               {_,_}
#    |\  ||  |\                                                |
#    | | ||  | |              Github link:                   (\|  /)
#    | | || / /         <github.com/mm22b011-deepaks>         \| //
#     \ \||/ /                                                 |//
#      `\\//`   \\   \./    \\ /     //    \\./   \\   //   \\ |/ /
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#####################################################################

########################  IMPORTANT  ################################
# Download the a file from the given link pasted here for it to work

# https://drive.google.com/file/d/1OvWAiAxAIXmww4Eou_vutsxElzdf0cPV/view?usp=sharing
########################  IMPORTANT  ################################
import numpy as np


def s_box_conversion(byte):
    
    s_box = [
        0x65, 0x4c, 0x6a, 0x42, 0x4b, 0x63, 0x43, 0x6b, 0x55, 0x75, 0x5a, 0x7a, 0x53, 0x73, 0x5b, 0x7b,
    0x35, 0x8c, 0x3a, 0x81, 0x89, 0x33, 0x80, 0x3b, 0x95, 0x25, 0x98, 0x2a, 0x90, 0x23, 0x99, 0x2b,
    0xe5, 0xcc, 0xe8, 0xc1, 0xc9, 0xe0, 0xc0, 0xe9, 0xd5, 0xf5, 0xd8, 0xf8, 0xd0, 0xf0, 0xd9, 0xf9,
    0xa5, 0x1c, 0xa8, 0x12, 0x1b, 0xa0, 0x13, 0xa9, 0x05, 0xb5, 0x0a, 0xb8, 0x03, 0xb0, 0x0b, 0xb9,
    0x32, 0x88, 0x3c, 0x85, 0x8d, 0x34, 0x84, 0x3d, 0x91, 0x22, 0x9c, 0x2c, 0x94, 0x24, 0x9d, 0x2d,
    0x62, 0x4a, 0x6c, 0x45, 0x4d, 0x64, 0x44, 0x6d, 0x52, 0x72, 0x5c, 0x7c, 0x54, 0x74, 0x5d, 0x7d,
    0xa1, 0x1a, 0xac, 0x15, 0x1d, 0xa4, 0x14, 0xad, 0x02, 0xb1, 0x0c, 0xbc, 0x04, 0xb4, 0x0d, 0xbd,
    0xe1, 0xc8, 0xec, 0xc5, 0xcd, 0xe4, 0xc4, 0xed, 0xd1, 0xf1, 0xdc, 0xfc, 0xd4, 0xf4, 0xdd, 0xfd,
    0x36, 0x8e, 0x38, 0x82, 0x8b, 0x30, 0x83, 0x39, 0x96, 0x26, 0x9a, 0x28, 0x93, 0x20, 0x9b, 0x29,
    0x66, 0x4e, 0x68, 0x41, 0x49, 0x60, 0x40, 0x69, 0x56, 0x76, 0x58, 0x78, 0x50, 0x70, 0x59, 0x79,
    0xa6, 0x1e, 0xaa, 0x11, 0x19, 0xa3, 0x10, 0xab, 0x06, 0xb6, 0x08, 0xba, 0x00, 0xb3, 0x09, 0xbb,
    0xe6, 0xce, 0xea, 0xc2, 0xcb, 0xe3, 0xc3, 0xeb, 0xd6, 0xf6, 0xda, 0xfa, 0xd3, 0xf3, 0xdb, 0xfb,
    0x31, 0x8a, 0x3e, 0x86, 0x8f, 0x37, 0x87, 0x3f, 0x92, 0x21, 0x9e, 0x2e, 0x97, 0x27, 0x9f, 0x2f,
    0x61, 0x48, 0x6e, 0x46, 0x4f, 0x67, 0x47, 0x6f, 0x51, 0x71, 0x5e, 0x7e, 0x57, 0x77, 0x5f, 0x7f,
    0xa2, 0x18, 0xae, 0x16, 0x1f, 0xa7, 0x17, 0xaf, 0x01, 0xb2, 0x0e, 0xbe, 0x07, 0xb7, 0x0f, 0xbf,
    0xe2, 0xca, 0xee, 0xc6, 0xcf, 0xe7, 0xc7, 0xef, 0xd2, 0xf2, 0xde, 0xfe, 0xd7, 0xf7, 0xdf, 0xff,
    ]

    
    if 0 <= byte < len(s_box):
        return s_box[byte]
    else:
        raise ValueError("Byte value out of range for S-Box")




LFSR_8_TK2 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 193, 195, 197, 199, 201, 203, 205, 207, 209, 211, 213, 215, 217, 219, 221, 223, 225, 227, 229, 231, 233, 235, 237, 239, 241, 243, 245, 247, 249, 251, 253, 255, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183, 185, 187, 189, 191, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254]
LFSR_8_TK3 = [0, 128, 1, 129, 2, 130, 3, 131, 4, 132, 5, 133, 6, 134, 7, 135, 8, 136, 9, 137, 10, 138, 11, 139, 12, 140, 13, 141, 14, 142, 15, 143, 16, 144, 17, 145, 18, 146, 19, 147, 20, 148, 21, 149, 22, 150, 23, 151, 24, 152, 25, 153, 26, 154, 27, 155, 28, 156, 29, 157, 30, 158, 31, 159, 160, 32, 161, 33, 162, 34, 163, 35, 164, 36, 165, 37, 166, 38, 167, 39, 168, 40, 169, 41, 170, 42, 171, 43, 172, 44, 173, 45, 174, 46, 175, 47, 176, 48, 177, 49, 178, 50, 179, 51, 180, 52, 181, 53, 182, 54, 183, 55, 184, 56, 185, 57, 186, 58, 187, 59, 188, 60, 189, 61, 190, 62, 191, 63, 64, 192, 65, 193, 66, 194, 67, 195, 68, 196, 69, 197, 70, 198, 71, 199, 72, 200, 73, 201, 74, 202, 75, 203, 76, 204, 77, 205, 78, 206, 79, 207, 80, 208, 81, 209, 82, 210, 83, 211, 84, 212, 85, 213, 86, 214, 87, 215, 88, 216, 89, 217, 90, 218, 91, 219, 92, 220, 93, 221, 94, 222, 95, 223, 224, 96, 225, 97, 226, 98, 227, 99, 228, 100, 229, 101, 230, 102, 231, 103, 232, 104, 233, 105, 234, 106, 235, 107, 236, 108, 237, 109, 238, 110, 239, 111, 240, 112, 241, 113, 242, 114, 243, 115, 244, 116, 245, 117, 246, 118, 247, 119, 248, 120, 249, 121, 250, 122, 251, 123, 252, 124, 253, 125, 254, 126, 255, 127]
permutation = [9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7]

Inverse_Lookup_Table = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 193, 195, 197, 199, 201, 203, 205, 207, 209, 211, 213, 215, 217, 219, 221, 223, 225, 227, 229, 231, 233, 235, 237, 239, 241, 243, 245, 247, 249, 251, 253, 255, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183, 185, 187, 189, 191, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254]
KEY = [0] * 16

new_file_path = 'test_trace-pt.npy'  
data = np.load(new_file_path)
total_traces = len(data)

hex_data = np.array([[np.array([format(byte, '02x') for byte in plaintext]) for plaintext in pair] for pair in data])


power_traces_file_path = 'test_trace-ct.npy'  


final_results = np.zeros((total_traces, 256), dtype=np.uint8)  
final_results_2 = np.zeros((total_traces, 256), dtype=np.uint8)

constant_matrix = [
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [2, 0, 0, 0],
    [0, 0, 0, 0]
]


TK1 = [
    [2, 0, 0, 0],
    [0, 0, 0, 26],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]

constant_matrix_2 = [
    [3, 0, 0, 0],
    [0, 0, 0, 0],
    [2, 0, 0, 0],
    [0, 0, 0, 0]
]


correct_points = []
correct_points_2 = []
key_guess_matrix = np.zeros((4, 4), dtype=int)
key_guess_matrix_2 = np.zeros((4, 4), dtype=int)

s_box_value = s_box_conversion(00)
###################################### ROUND 1 ATTACK ######################################################
for byte_index in range(8):

    
    for trace_index in range(total_traces):
        
        first_plaintext = hex_data[trace_index]  
        matrix1 = first_plaintext[0]  
        matrix2 = first_plaintext[1]  
        
        byte_from_first_row = int(matrix1[byte_index], 16)  
        byte_from_second_row = int(matrix2[0], 16)  

        constant_index_row = byte_index // 4
        constant_index_col = byte_index % 4
        tk1_row = byte_index // 4
        tk1_col = byte_index % 4

        
        for key in range(256): 
            result_after_first_xor = s_box_value ^ constant_matrix[constant_index_row][constant_index_col]
            final_result = result_after_first_xor ^ byte_from_first_row ^ key ^ TK1[constant_index_row][constant_index_col]
            final_results[trace_index][key] = final_result

    
    for i in range(final_results.shape[0]):  
        for j in range(final_results.shape[1]):  
            final_results[i][j] = bin(final_results[i][j]).count('1')  

    hamming_weights = final_results
    power_traces = np.load(power_traces_file_path)  
    correlation_array = np.zeros((256, power_traces.shape[1]))
    power_traces_mean = power_traces.mean(axis=0)
    power_traces_std = power_traces.std(axis=0)
    normalized_power_traces = (power_traces - power_traces_mean) / power_traces_std

    for key in range(256):
        hamming_set = hamming_weights[:, key] 
        hamming_set_mean = np.mean(hamming_set)
        hamming_set_std = np.std(hamming_set)
        normalized_hamming_set = (hamming_set - hamming_set_mean) / hamming_set_std
        
        correlation = np.dot(normalized_hamming_set, normalized_power_traces) / total_traces
        correlation_array[key, :] = correlation

 
    max_correlation = -np.inf
    max_point = None
    max_key_guess = None
    
   
    if byte_index == 0:
        search_start = 4000
    else:
        search_start = correct_points[0] + byte_index*67
    search_end = search_start + 1000
 
    for key_guess in range(256):
        for point in range(search_start, search_end):  
            correlation_value = correlation_array[key_guess, point]
            
            if correlation_value > max_correlation:
                max_correlation = correlation_value
                max_point = point
                max_key_guess = key_guess

        
    correct_points.append(max_point)
    
    row = byte_index // 4  
    col = byte_index % 4   

    
    key_guess_matrix[row, col] = max_key_guess
    KEY[byte_index] = max_key_guess
########################################################################################

mix_matrix = np.zeros((4, 4), dtype=int)

result_matrices = []
mix_matrices = []

for plaintext_index in range(total_traces):
    
    first_plaintext = hex_data[plaintext_index]  
    mat1 = first_plaintext[0]  
    mat2 = first_plaintext[1]  
    matrix1_2d = [mat1[i:i+4] for i in range(0, 16, 4)]
    
    mat1_value = [[0] * 4 for _ in range(4)]  

    for i in range(4):
        for j in range(4):
            
            mat1_value[i][j] = int(matrix1_2d[i][j], 16)
    
    result_matrix = np.zeros((4, 4), dtype=int)

    for i in range(2):  
        for j in range(4):  
            constant_value = int(constant_matrix[i][j])
            TK1_value = int(TK1[i][j])
            key_guess_value = int(key_guess_matrix[i][j])
            matrix1_value_element  = mat1_value[i][j]
            s_box_value_int= int(s_box_value)
            xor_result = constant_value ^ TK1_value ^ key_guess_value ^ matrix1_value_element  ^ s_box_value_int
            result_matrix[i][j] = xor_result

    for i in range(2, 4):  
        for j in range(4): 
            result_matrix[i][j] = int(constant_matrix[i][j]) ^ int(s_box_value)

    result_matrices.append(result_matrix)

result_matrices = np.array(result_matrices)

def rotate_row(row, positions):
    positions %= len(row)  
    return np.roll(row, positions)

for plaintext_index in range(total_traces):

    for i in range(4):
        result_matrices[plaintext_index][i] = rotate_row(result_matrices[plaintext_index][i], i)

for plaintext_index in range(total_traces):
    mix_result = result_matrices[plaintext_index]
    mix_result_flat = mix_result.flatten()
    for j in range(4):
        mix_result_flat[j], mix_result_flat[4+j], mix_result_flat[8+j], mix_result_flat[12+j] = mix_result_flat[j] ^ mix_result_flat[8+j] ^ mix_result_flat[12+j], mix_result_flat[j], mix_result_flat[4+j] ^ mix_result_flat[8+j], mix_result_flat[0+j] ^ mix_result_flat[8+j]
    mix_result = mix_result_flat.reshape(4, 4)
    mix_matrices.append(mix_result_flat)
mix_matrices = np.array(mix_matrices)

sets_of_matrices = np.random.randint(0, 256, size=(total_traces, 3, 4, 4))

TK_result = np.zeros((total_traces, 48), dtype=np.uint8)
TK_result_rep = np.zeros((total_traces, 48), dtype=np.uint8)
TK1 = np.array(TK1)

for i in range(total_traces):
    plaintext = data[i]  
    tk2 = plaintext[0]
    tk2 = np.array([int(x) for x in tk2])
    key_guess = np.array(key_guess_matrix) 
    if TK1.size != 16 or tk2.size != 16 or key_guess.size != 16:
        raise ValueError(f"Incorrect size of matrices: TK1={TK1.size}, tk2={tk2.size}, key_guess={key_guess.size}")
    TK_result[i] = np.concatenate([TK1.flatten(), tk2.flatten(), key_guess.flatten()])

########################################### 2nd Round #####################################################

TK_2_result = np.zeros((total_traces, 48), dtype=np.uint8)

all_tk1 = []
all_tk2 = []
all_tk3 = []

TK_2 = np.zeros((1, 48), dtype=int)  
for i in range(total_traces):
    TK_2 = np.copy(TK_result[i])
    
    for j in range(48): TK_2[j] = TK_result[i][j-j%16+permutation[j%16]]
    for j in range(8):
        TK_2[j+16] = LFSR_8_TK2[TK_2[j+16]]
        TK_2[j+32] = LFSR_8_TK3[TK_2[j+32]]
        
        TK_2[8+j] = TK_result[i][j]
        TK_2[24+j] = TK_result[i][16+j]
    TK_2_result[i] = TK_2
    
    tk1 = TK_2[:16]
    tk2 = TK_2[16:32]
    tk3 = TK_2[32:]

    
    all_tk1.append(tk1)
    all_tk2.append(tk2)
    all_tk3.append(tk3)

##################################Tweak Key Computaion end #######################################

for byte_index_2 in range(8):

    
    for trace_index_2 in range(total_traces):
        constant_index_row_2 = byte_index_2 // 4
        constant_index_col_2 = byte_index_2 % 4
        matrix1_2 = all_tk2[trace_index_2]  
        byte_from_second_row_2 = int(mix_matrices[trace_index_2][byte_index_2])  
        byte_from_first_row_2 = matrix1_2[byte_index_2] 
        s_box_value_2 = s_box_conversion(byte_from_second_row_2)

        for key_2 in range(256):  
            result_after_first_xor_2 = s_box_value_2 ^ constant_matrix_2[constant_index_row_2][constant_index_col_2]
            final_result_2 = result_after_first_xor_2 ^ byte_from_first_row_2 ^ key_2 ^ all_tk1[trace_index_2][byte_index_2]
            final_results_2[trace_index_2][key_2] = final_result_2

    for i in range(final_results_2.shape[0]):  
        for j in range(final_results_2.shape[1]):  
            final_results_2[i][j] = bin(final_results_2[i][j]).count('1')  

    hamming_weights_2 = final_results_2
    power_traces = np.load(power_traces_file_path)  
    correlation_array = np.zeros((256, power_traces.shape[1]))
    power_traces_mean = power_traces.mean(axis=0)
    power_traces_std = power_traces.std(axis=0)
    normalized_power_traces = (power_traces - power_traces_mean) / power_traces_std

    
    for key_2 in range(256):
        
        hamming_set_2 = hamming_weights_2[:, key_2]  
        hamming_set_mean_2 = np.mean(hamming_set_2)
        hamming_set_std_2 = np.std(hamming_set_2)
        normalized_hamming_set_2 = (hamming_set_2 - hamming_set_mean_2) / hamming_set_std_2
        correlation_2 = np.dot(normalized_hamming_set_2, normalized_power_traces) / total_traces
        correlation_array[key_2, :] = correlation_2  


    max_correlation = -np.inf
    max_point = None
    max_key_guess = None
    max_key_guess_indexed = None
    
    
    if byte_index_2== 0:
        search_start_2 = 12000
    else:
        search_start_2 = correct_points_2[0] + byte_index_2*67
    
    search_end_2 = search_start_2 + 1000
    
    
    for key_guess in range(256):
        for point in range(search_start_2, search_end_2):  
            correlation_value = correlation_array[key_guess, point]
            
            if correlation_value > max_correlation:
                max_correlation = correlation_value
                max_point = point
                max_key_guess = key_guess
                max_key_guess_indexed = Inverse_Lookup_Table[max_key_guess]
        
    correct_points_2.append(max_point)

    KEY[permutation[byte_index_2]] = max_key_guess_indexed

print("KEY :")

key_output = ''.join([f"{value:02x}" for byte_index, value in enumerate(KEY)])
print(key_output)
  
