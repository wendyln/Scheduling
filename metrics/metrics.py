################ Set up ################

slots = list(np.sort(np.unique(sched['slot'].values)))

# Translation to different slot notation
slots_n = range(1,len(slots)+1)
d = dict(zip(slots, slots_n))


################ Calculate starting slots ################
def get_metrics():
  quint_start = []
  quad_start = []
  four_in_five_start = []  # not including quints or quads  
  triple_24_start = []
  triple_day_start = []
  three_in_four_start = []  # not including triples
  eve_morn_start = []
  other_b2b_start = []
  two_in_three_start = []
  
  slots_e = slots + [np.inf]*10
  
  for j in range(len(slots)):
      s = slots[j]
      if s+1 == slots_e[j+1]: # 11  
          if s%slots_per_day == 0:
              eve_morn_start.append(d[s])
          else:
              other_b2b_start.append(d[s])       
          if s+2 == slots_e[j+2]: # 111
              if slots_per_day - s%slots_per_day >= 2 and slots_per_day - s%slots_per_day != slots_per_day:
                  triple_day_start.append(d[s])
              else:
                  triple_24_start.append(d[s])             
              if s+3 == slots_e[j+3]: # 1111
                  quad_start.append(d[s])     
                  if s+4 == slots_e[j+4]: # 11111
                      quint_start.append(d[s])         
              else: #1110
                  three_in_four_start.append(d[s])
                  if s+4 == slots_e[j+3]: # 11101
                      four_in_five_start.append(d[s]) 
          else: # 110
              if s+3 == slots_e[j+2]: # 1101
                  three_in_four_start.append(d[s])        
                  if s+4 == slots_e[j+3]: # 11011
                      four_in_five_start.append(d[s])                
      else: # 10 
          if s+2 == slots_e[j+1]: # 101
              two_in_three_start.append(d[s])     
              if s+3 == slots_e[j+2]: #1011
                  three_in_four_start.append(d[s])     
                  if s+4 == slots_e[j+3]: #10111
                      four_in_five_start.append(d[s])
                      
  # Adjust two_in_three_start
  n = []
  for j in two_in_three_start:
      n.append([j,j+1])
  for j in triple_24_start + triple_day_start:
      n.append([j,j+2])
  two_in_three_start = np.sort(np.array(n), axis=0)

  metrics = {
    quint = "quint_start", 
    quad = "quad_start", 
    four_in_five = "four_in_five_start", 
    triple_24 = "triple_24_start", 
    triple_day = "triple_day_start", 
    three_in_four = "three_in_four_start", 
    eve_morn = "eve_morn_start", 
    two_in_three = "two_in_three_start", 
  }
  return metrics

