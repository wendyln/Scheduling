import gurobipy as gp
from gurobipy import *
from itertools import product


# Generate Indexing Set
group_pairs = []
for i in exam_groups:
    for j in exam_groups:
        if group_coenroll_matrix.at[i,j] > 0:
            group_pairs.append((i,j))
            


def scheduling_IP(alpha, beta, gamma, t, p, blocks, slots = None, slots_to_skip = None):  
    ''' alpha := weight penalty for 3 exams within the same day
        beta := weight penalty for 3 exams within 24 hours
        gamma := weight penalty for back-to-back exams
        t := triplet coenrollment dictionary (t[(i,j,k)] is the mutual coenrollment in blocks i,j,k)
        p := pairwise coenrollment dictionary (p[(i,j)] is the coenrollment in blocks i and j)
        blocks := array or list [1, ..., n] representing distinct blocks from the block assignment 
        slots := list of all available slots
        slots_to_skip := slots not to be used for scheduling
        '''
    if not slots:
        slots = blocks
    
    # Indexing for decision variables
    block_sequence_slot = []
    block_sequence = []
    for i in blocks_d:
        for j in blocks_d:
            for k in blocks_d:
                block_sequence.append((i,j,k))
                for s in slots:
                    block_sequence_slot.append((i,j,k,s))


    shifted_slots = np.roll(slots, -1)
    next_slot = dict(zip(slots, shifted_slots))

    # Start slot indexing for objective
    triple_slots = slots[:-2]
    triple_in_day = np.arange(1, len(triple_slots)+1, 3)
    triple_in_24hr = np.array(list(set(triple_slots).difference(set(triple_in_day))))
    b2b = np.arange(1, len(slots))

    m = Model('Scheduler')
    x = m.addVars(block_sequence_slot, vtype = GRB.BINARY, name='group_seq_indicator')
    
    triple_in_day_var = m.addVar(vtype = GRB.INTEGER, name='triple_in_day')
    triple_in_24hr_var = m.addVar(vtype = GRB.INTEGER, name='triple_in_24hr')
    b2b_var = m.addVar(vtype = GRB.INTEGER, name='back_to_back')
    
    schedule = m.addVars(slots, vtype = GRB.INTEGER, name='slot_assignment')

    # Each group appears as the first of a triple exactly once
    m.addConstrs((gp.quicksum(x[(i,j,k,s)] for j in blocks_d for k in blocks_d for s in slots) == 1 for i in blocks),
                 name='each_group_scheduled_i')

    # Each group appears as the second of a triple exactly once
    m.addConstrs((gp.quicksum(x[(i,j,k,s)] for i in blocks_d for k in blocks_d for s in slots) == 1 for j in blocks),
                 name='each_group_scheduled_j')

    # Each group appears as the third of a triple exactly once
    m.addConstrs((gp.quicksum(x[(i,j,k,s)] for i in blocks_d for j in blocks_d for s in slots) == 1 for k in blocks),
                 name='each_group_scheduled_k')

    # Each slot is assigned to exactly one group
    m.addConstrs((gp.quicksum(x[(i,j,k,s)] for i in blocks_d for j in blocks_d for k in blocks_d) == 1 for s in slots),
                 name='each_slot_one_group')

    # Prevent repeats in group sequences
    m.addConstrs((x[(i,i,k,s)] == 0 for i in blocks for k in blocks for s in slots), name='degenerate_ij')
    m.addConstrs((x[(i,j,i,s)] == 0 for i in blocks for j in blocks for s in slots), name='degenerate_ik')
    m.addConstrs((x[(i,j,j,s)] == 0 for i in blocks for j in blocks for s in slots), name='degenerate_jk')

        
    # Make sure block 20 goes to the fixed slots
    m.addConstrs((gp.quicksum(x[(20,j,k,s)] for j in blocks_d for k in blocks_d) == 1 for s in slots_to_skip), name='fixed_slots')
    
    # Latecomers (block 19) go to saturday afternoon (slot 22)
    m.addConstr((gp.quicksum(x[(19,j,k,22)] for j in blocks_d for k in blocks_d) == 1), name='latecomers')
    
    # Triplet slot continuity constraints (e.g. if x[1,2,3,1] = 1, then x[2,3,k,2] = 1 for some block k)
    m.addConstrs((
        gp.quicksum(x[(i,j,k,s)] for i in blocks_d) == gp.quicksum(x[(j,k,l,next_slot[s])] for l in blocks_d) 
        for j in blocks_d 
        for k in blocks_d for s in slots), name='continuity_constraints')
    
    # Limit 3_in_24hrs and 3_in_day
    m.addConstr(triple_in_day_var <= 60,name='triple_day_leq_60')
    m.addConstr(triple_in_24hr_var <= 125, name ='triple_24hr_leq_125')

    # Output
    m.addConstrs((schedule[s] == gp.quicksum(i*x[(i,j,k,s)] for i in blocks_d for j in blocks_d for k in blocks_d) 
                  for s in slots), 
                 name='write_output')
    m.addConstr((gp.quicksum(t[(i,j,k)]*x[(i,j,k,s)] for i in blocks_d for j in blocks_d for k in blocks_d 
                                 for s in list(triple_in_day)) == triple_in_day_var), name='write_penalty1')
    m.addConstr((gp.quicksum(t[(i,j,k)]*x[(i,j,k,s)] for i in blocks_d for j in blocks_d for k in blocks_d
                                 for s in list(triple_in_24hr)) == triple_in_24hr_var), name='write_penalty2')
    m.addConstr((gp.quicksum(p[(i,j)]*x[(i,j,k,s)] for i in blocks_d for j in blocks_d for k in blocks_d
                                 for s in list(b2b)) == b2b_var), name='write_penalty3')

    # Set Objective function
    m.setObjective(
                    alpha*gp.quicksum(t[(i,j,k)]*x[(i,j,k,s)] for i in blocks_d for j in blocks_d for k in blocks_d 
                                 for s in list(triple_in_day)) + 
                   beta*gp.quicksum(t[(i,j,k)]*x[(i,j,k,s)] for i in blocks_d for j in blocks_d for k in blocks_d
                                 for s in list(triple_in_24hr)) +
                   gamma*gp.quicksum(p[(i,j)]*x[(i,j,k,s)] for i in blocks_d for j in blocks_d for k in blocks_d
                                 for s in list(b2b)),
                   GRB.MINIMIZE)


    m.optimize()

    output = pd.DataFrame(columns = ['slot', 'block'])
    for k in schedule.keys():
        output = output.append({'slot':k, 'block':schedule[k].x}, ignore_index = True)
    output['slot'] = output['slot'].astype(int)
    obj = m.getObjective().getValue()
    penalty = {'triple_in_day': triple_in_day_var.x,
               'triple_in_24hr_var': triple_in_24hr_var.x,
               'back_to_back': b2b_var.x}
    
    return obj, output, penalty
