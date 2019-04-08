# Concave Hull

This toolkit generates a concave hull by starting from the convex hull and breaking the edges into two edges by looking for the closest
point to the edge that would form a wide enough angle and the new edges don't intersect with the existing edges.

## Parameters
1. A point feature class that serves as input
2. Optionally, a group by field in the point feature class. If provided, a polygon will be created for each unique value in that field.
3. The output feature class
4. A maximum number of iterations. This can be used to limit the number of edges created and is useful if fine-grained detail is not needed for a very large number of potential boundary points. If set to 0, the convex polygon will be generated.
5. A minimum angle (in degrees). The angle between two new edges will have to be at least this large. If set to 180, the convex polygon will be generated regardless of other settings.
6. A minimum side length (in fractions). The number provided is multiplied by the average edge length in the convex polygon. Specifying 1 means that any edges smaller than the average edge length of the convex polygon will not be further subdivided. Again, this is useful to quickly generate a coarser polygon.
