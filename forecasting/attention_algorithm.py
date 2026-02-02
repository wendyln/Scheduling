def similarity(query, keys, k, sem): 
    num_sem , num_classes = query.shape 
    batch, num_sem, num_classes = keys.shape 
    
    transpose_keys = keys.transpose(1,2)
    result = torch.matmul(query, transpose_keys)
    score = torch.sum(torch.diagonal(result, dim1=-2, dim2=-1) , 1) #1,  batch shape  or [batch]

    top_k_indices = score.argsort(descending=True)[:k]

    raw_score = score[top_k_indices]
    
    #preenroll similarity 
    value_preenroll = keys[top_k_indices][:,sem,:] # returns a matrix of k, # classes ( preenroll)
    
    value_post_add = keys[top_k_indices][:,sem + 1 ,:]# returns a matrix of k, # classes ( post add)

    return (raw_score, top_k_indices, value_preenroll, value_post_add) 

# n^2 runtime though ... 
# length of input_x = len(input_x)
def forward(data, input_x, k): 
    input_x['prediction'] = [[] for _ in range(len(input_x))]
    for std_idx in range(len(input_x)): 
        if std_idx % (len(input_x) // 10) == 0: 
            print("-------------------progress", std_idx * 100 // len(input_x), "% ----------------") 
   
        student = input_x['student'][std_idx]
        course = createVec(input_x['course'][std_idx])

        if student in data: 
            student_metadata = data[student] 
            sem = student_metadata['semesters']
            if sem > 5: 
                continue 
            student_matrix = student_metadata['matrix'] 

            key_filter = [student for student in data if data[student]['semesters'] > min(sem, 5)]
            keys = filter_matrix(key_filter)

            
            raw_score, top_k_indices, value_preenroll, value_post_add = similarity(student_matrix, keys, k, sem + 1)
            
        else : 
            key_filter = random.sample(student_data.keys(), 2*k)
            keys = filter_matrix(key_filter)
            value_preenroll = keys[:,0,:]
            value_post_add = keys[:,1,:]
            raw_score = torch.zeros(2* k)
        
        # calculate preenroll score 
        preenroll_score = preenroll_score_fn(course, value_preenroll)
        
        
        # score 
        score = raw_score + 1.5 * preenroll_score
        
        # softmax
        score = F.softmax(score, dim = 0)    
        value = score.unsqueeze(1) * value_post_add 
        score_value = value.sum(dim = 0)
        score_value = score_value + 0.5 * course

        pred = score_value.numpy()
        input_x.at[std_idx, 'prediction'] = pred
    return input_x

def iterative_scoring(x): 
    num_students = len(x) 
    for i in range(num_students): 
        student = x['student'][i]
        current_course = createVec(x['course'][i])
        if student in student_data: # not in database : freshman 
            student_data[student]['matrix'][:, student_data[student]['last i'] + 1 ] = current_course
            student_sem = x['student']['semesters'] + 1 
            similar_students, similarityscore = find_top_similar_students(student, student_data) 
            
            
            
def similarity_score_iter(query, key, top_n=200):
    
    """
    Query: (number of class, number of students)
    Keys : (number of class, number of key students)
    
    Q^TK --> aij is similarity of query i and key j 
    Return similarity score and filtered index / filtered studentid 
    """
    
    # Calculate similarity matrix
    similarity_matrix = np.dot(query, key.T)  # Similarity score matrix 
    top_indices = np.argsort(similarity_matrix, axis=1)[:, -top_n:]
    
    return top_indices

def find_top_similar_students(student_id, student_data, similarity_threshold=10):
    # Get the matrix of classes attended by the given student
    student_matrix = student_data[student_id]['matrix']

    # Initialize a dictionary to store the top similar students and their similarity scores for each semester
    top_similar_students = {semester: {'students': [], 'scores': []} for semester in range(16)}

    # Initially, consider all students for comparison
    filtered_indices = list(student_data.keys())

    # Iterate over each semester
    for semester in range(16):
        # Get the matrix of classes attended by all students for the current semester
        all_students_matrix = np.array([student_data[s_id]['matrix'][:, semester] for s_id in filtered_indices])

        # Calculate similarity with all other students for the current semester
        similarities = np.dot(student_matrix[:, semester], all_students_matrix.T) 

        # Filter students based on similarity score threshold
        filtered_indices = [idx for idx, sim in zip(filtered_indices, similarities) if sim > similarity_threshold]
        
        # Store the filtered similar students and their similarity scores for the current semester
        for idx, sim in zip(filtered_indices, similarities):
            if idx in top_similar_students[semester]['students']:
                # If the similar student is already in top_similar_students, accumulate the score
                score_index = top_similar_students[semester]['students'].index(idx)
                top_similar_students[semester]['scores'][score_index] += sim
            else:
                # Otherwise, add the student and score to top_similar_students
                top_similar_students[semester]['students'].append(idx)
                top_similar_students[semester]['scores'].append(sim)


    return top_similar_students
  
def iterative_scoring(student_df, student_data, threshold = 10):
    num_students = len(student_df)
    for i in range(num_students):
        student = student_df['student'][i]
        current_course = createVec(student_df['course'][i])
        if student in student_data: # If student exists in database
            last_semester = student_data[student]['last_semester']
            student_data[student]['matrix'][:, last_semester + 1] = current_course
            student_sem = last_semester + 1
        else:  # If student is a freshman
            student_data[student] = {'matrix': np.zeros((len(all_courses), 16)), 'last_semester': -1}
            student_data[student]['matrix'][:, 0] = current_course
            student_sem = 0

        similar_students = find_top_similar_students(student, student_data, threshold)
        
        # Initialize predicted enrollment vector for the current student
        predicted_enrollment = np.zeros((len(all_courses), 1))

        for semester, similar_students_info in similar_students.items():
            softmax_scores = softmax(similar_students_info['scores'])
            for similar_student, softmax_score in zip(similar_students_info['students'], softmax_scores):
                predicted_enrollment += softmax_score * student_data[similar_student]['matrix'][:, student_sem + 1]

        # Add predicted enrollment to student_df
        student_df.at[i, 'prediction'] = predicted_enrollment

        # Update last semester for the current student
        student_data[student]['last_semester'] = student_sem + 1

    return student_df

def main(): 
  # prediction = forward(student_data, x_preenroll, 20)
  prediction = iterative_scoring(x_preenroll[:10], student_data, 10)
  return prediction 
        
