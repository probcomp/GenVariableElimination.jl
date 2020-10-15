# GenVariableElimination.jl


# sampling and joint probability given variable elimination sequence - 
# in reverse elimination order:
# - we have a partial assignment to all variables that came after the variable in the elimination ordering
# - look up the appropriate intermediate factor graph, which is the FG immediately before eliminating the variable
#   identify all factors in this intermediate FG that are connected to this variable
#   take the relevant slices for each factor, resulting in vectors, and point-wise multiply them
#   normalize, and then sample from this distribution, and add the value of the sample to the partial assignment

# joint probability given variable elimination sequence and full assignment
# - do the same process as above, but taking values instead of sampling them

# constructor from trace and addr info (queries trace with update)

