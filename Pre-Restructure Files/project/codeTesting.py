import indices as indices
from csvToPandas import csvToPandas


data = csvToPandas('sample')
print(indices.pDeltaPavg1(data))
print(indices.pDeltaPavg2(data))
