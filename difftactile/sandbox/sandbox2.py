skip = x
x = GINEConv(x, edge_index, edge_attr)
x = BatchNorm(x)
x = ReLU(x)
x = Dropout(x)
x = x + skip
