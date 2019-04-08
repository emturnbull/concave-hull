"""

    Author:                Erin Turnbull
    Last Modified:        2019-03-29
    
    Provides various geometric functions for point/line/polygon operations. Focuses on concave, convex hull and point in polygon.
"""

import math
import tweet
import sys
import heapq

class ExtendedTupleList:
    """
        Extends a list of tuples to provide additional functionality. Notable additions include:
        
        - The ability to extend() a list without adding two duplicate items in a row. Non-sequential duplicates are allowed.
        - The ability to reverse() only a portion of the list
        - A swap() method
        - A ranges() method that provides the min/max values in any particular index of the tuple.
        - A ranges_within() method that provides the min/max values of particular indexes subgrouped by other indexes.
        - The ability to generate a sublist using filter().
        - An in-place quicksort() based on any particular key index.
        - sub_quicksort() maintains blocks based on one sorted key index and sorts those blocks based on another key.
        - double_sort() combines quicksort() and sub_quicksort() to sort by one key, then another key within those blocks.
    """

    # Slots is used to make this faster.
    __slots__ = ['list']
    
    def __init__(self, list: list):
        """ Initializes this class. """
        self.list = list
        
    def __len__(self):
        """ Delegate len() to the list """
        return len(self.list)
        
    def __getitem__(self, key):
        """ Delegate list access to the list """
        return self.list[key]
    
    def __setitem__(self, key, value):
        """ Delegate item setting to the list """
        self.list[key] = value
    
    def __delitem__(self, key):
        """ Delegate deletion to the list """
        del self.list[key]
        
    def __str__(self):
        """ Delegate str() typecast to the list """
        return str(self.list)
        
    def append(self, value):
        """ Delegate append() to the list """
        self.list.append(value)
        
    def insert(self, index, value):
        """ Delegate insert() to the list """
        self.list.insert(index, value)
        
    def pop(self):
        """ Delegate pop() to the list """
        self.list.pop()
        
    def extend(self, other_list:list, avoid_repeats:bool=False):
        """ If avoid_repeats is False, delegates extend() to the list. Otherwise, appends all items that don't create a repeat of 2 items to the list."""
        if not avoid_repeats:
            self.list.extend(other_list)
        else:
            for item in other_list:
                if not self.list or not self.list[-1] == item:
                    self.list.append(item)
    
    def reverse(self, start:int=0, end:int=None):
        """ Reverses the portion of the list between start and end indexes, inclusive. """
        if end == None:
            if start == 0:
                self.list.reverse()
                return
            end = len(self) - 1
        left = start
        right = end
        while left < right:
            self.swap(left, right)
            left += 1
            right -= 1
        
    def swap(self, index_a:int, index_b:int):
        """ Swaps two items in the list."""
        if not index_a == index_b:
            self.list[index_a], self.list[index_b] = self.list[index_b], self.list[index_a]
    
    def ranges(self, keys:list)->list:
        """ Determines the minimum and maximum values for any particular tuple index within the list.
            
            Returns
            -------
            list:
                A list with keys matching the keys parameter. Each entry contains a two-element list,
                where [0] is the minimum and [1] is the maximum.
        """
        if not isinstance(keys, list):
            keys = [keys]
        ranges = {}
        for key in keys:
            ranges[key] = [None, None]
        for list_item in self.list:
            for key in keys:
                if ranges[key][0] is None:
                    ranges[key][0] = list_item[key]
                    ranges[key][1] = list_item[key]
                elif ranges[key][0] > list_item[key]:
                    ranges[key][0] = list_item[key]
                elif ranges[key][1] < list_item[key]:
                    ranges[key][1] = list_item[key]
        return ranges
        
    def range_within(self, key_a:int, key_b:int)->list:
        """ Determines two-dimensional extrema.
            
            In essence, this algorithm finds four sets of items: those with a minimum or maximum value
            for key_a, as well as key_b. Then, within those sets, it determines the extrema of the other 
            key.
            
            For example, if you wanted to determine the range of X coordinates along the top and bottom of
            a set of points, you could use this method.
            
            Returns
            -------
            dict:
                A dictionary with the following structure:
                1. key_a or key_b (the "outside" key). The "inside" key is the other one.
                2. "min" or "max" (so whether we are dealing with the minima or maxima of the outside key)
                3. 0, 1, or 2: 
                  - 0 is the value of the outside key extrema
                  - 1 is the minimum value of the inside key of tuples that have an outside key equal to the value in 0
                  - 2 is the maximum value of the inside key under the same conditions.
        """
        ranges = {key_a: {"min": [None, None, None], "max": [None, None, None]}, key_b: {"min": [None, None, None], "max": [None, None, None]}}
        action_items = [(key_a, key_b, "min"), (key_a, key_b, "max"), (key_b, key_a, "min"), (key_b, key_a, "max")]
        for list_item in self.list:
            for action in action_items:
                key_o, key_i, ext = action
                if ranges[key_o][ext][0] == None or \
                    (ext == "min" and ranges[key_o][ext][0] > list_item[key_o]) or \
                    (ext == "max" and ranges[key_o][ext][0] < list_item[key_o]):
                    ranges[key_o][ext] = [list_item[key_o], list_item[key_i], list_item[key_i]]
                elif ranges[key_o][ext][0] == list_item[key_o]:
                    if ranges[key_o][ext][1] > list_item[key_i]:
                        ranges[key_o][ext][1] = list_item[key_i]
                    elif ranges[key_o][ext][2] < list_item[key_i]:
                        ranges[key_o][ext][2] = list_item[key_i]
        return ranges
        
    def filter(self, filters:list)->list:
        """ A generator that filters through the tuples under specific conditions that can be specified.
            
            Parameters
            ----------
            filters:list
                A list of different filters. All of the conditions must be true at this time. Filters 
                are a tuple with three parts:
                
                - The tuple index that they operate on
                - The value that they use in their operation
                - The operation they perform (<, >, <=, >=, or = at the moment)
                
            Yields
            -------
            item:
                Provides items from the internal list that meet all of the filter conditions.
                
        """
        for item in self.list:
            use_item = True
            for filter in filters:
                filter_key, filter_value, filter_type = filter
                if filter_type == "<" and item[filter_key] >= filter_value:
                    use_item = False
                    break
                elif filter_type == ">" and item[filter_key] <= filter_value:
                    use_item = False
                    break
                elif filter_type == "<=" and item[filter_key] > filter_value:
                    use_item = False
                    break
                elif filter_type == ">=" and item[filter_key] < filter_value:
                    use_item = False
                    break
                elif filter_type == "=" and not item[filter_key] == filter_value:
                    use_item = False
                    break
            if use_item:
                yield item
                
    def double_sort(self, outside_key:int, inner_key:int, start:int=0, end:int=None, reverse_outside:bool=False, reverse_inside:bool=False):
        """ 
            Quicksorts the list by outside_key, then divides the list by stable blocks of outside_key
            and quicksorts those blocks by inner_key. Essentially equivalent to SQL statement of
            SORT BY outside_key, inner_key.
        """
        self.quicksort(outside_key, start, end)
        if reverse_outside:
            self.reverse(start, end)
        self.sub_quicksort(outside_key, inner_key, start, end, reverse_inside)
    
    def sub_quicksort(self, stable_key:int, sort_key:int, start:int=0, end:bool=None, reverse:bool=False):
        """ Quicksorts subsets of the list grouped by a stable key. In-place, non-recursive.
        
            This function maintains the order of blocks of tuples having the same stable_key. 
            Within that block, items are resorted by sort_key using quicksort(). Since
            quicksort() is ascending, specifying reverse = True will reverse the order within
            those blocks after quicksorting.
        """
        if end == None:    
            end = len(self) - 1
        if start >= end:
            return
        first = start
        for index in range(start + 1, end + 1):
            if not self[index][stable_key] == self[first][stable_key]:
                self.quicksort(sort_key, first, index - 1)
                if reverse:
                    self.reverse(first, index - 1)
                first = index
        if not first == end:
            self.quicksort(sort_key, first, end)
            if reverse:
                self.reverse(first, end)
    
    def quicksort(self, key:int, start:int=0, end:int=None):
        """ A non-recursive, in-place version of quicksort.
        
            Note that Python has not-great tail-recursion properties, so a recursive approach is
            not generally recommended. This is in-place to save on memory. Otherwise, it is a 
            straight-forward ascending quicksort of all the items between start and end indexes
            comparing the values in the "key" slot of the tuples in the list.
            
            There is a quick fall back if only two or fewer items need to be considered.
        """
        if end == None:
            end = len(self) - 1
        if start >= end:
            return
        if start == end - 1:
            if self[start][key] > self[end][key]:
                self.swap(start, end)
            return
        work = [(start, end)]
        while work:
            first, last = work.pop()
            pivot = (first + last) // 2
            self.swap(pivot, last)
            pivot_to = first
            for index in range(first, last + 1):
                if self[index][key] < self[last][key]:
                    self.swap(index, pivot_to)
                    pivot_to += 1
            self.swap(pivot_to, last)
            if pivot_to > (first + 1):
                work.append((first, pivot_to - 1))
            if last > (pivot_to + 1):
                work.append((pivot_to + 1, last))

def collinear(a:tuple, b:tuple, c:tuple)->bool:
    """ Determines if three points are collinear. """
    return ((b[1] - c[1]) * (a[0] - b[0])) == ((a[1] - b[1]) * (b[0] - c[0]))
        
def direction(a:tuple, b:tuple, c:tuple)->int:
    """ Determines whether the lines AB and BC make a counterclockwise or clockwise turn."""
    return ((b[1] - a[1]) * (c[0] - b[0])) - ((b[0] - a[0]) * (c[1] - b[1]))

def orientation(a:tuple, b:tuple, c:tuple)->int:
    """ Determine if CCW (1), CW(-1), or colinear(0) """
    d = direction(a, b, c)
    if d == 0:
        return 0
    elif d > 0:
        return 1
    else:
        return -1

def positive_slope(line:tuple)->bool:
    """ Determines if the slope of a line is positive."""
    return line[0][1] < line[1][1] == line[0][0] < line[1][0]
    
def is_upwards(line:tuple)->bool:
    """ Determines if a line moves up from left to right."""
    return line[1][1] > line[0][1]
    
def is_horizontal(line:tuple)->bool:
    """ Determines if a line is horizontal."""
    return line[0][1] == line[1][1]
            
def line_length_angle(line:tuple)->tuple:
    """ Determines the length and the cosine of the angle from a positive horizontal ray of a line segment."""
    squared_dist = point_sqr_distance(line[0], line[1])
    if squared_dist == 0:
        return 0,1
    distance = math.sqrt(squared_dist)
    angle_cosine = (line[1][0] - line[0][0]) / distance
    return squared_dist, angle_cosine
    
def edgify(vertices:list)->list:
    """ Takes a sequential list of vertices and turns it into a list of edges. """
    edges = []
    for k in range(0, len(vertices) - 1):
        edges.append([vertices[k], vertices[k + 1]])
    return edges

def closest_line_point(point:tuple, edge:tuple)->tuple:
    """ Determines the closest point on the infinite line associated with the edge to the given point.
    
        The closest point on an infinite line to a point is determined by the
        intersection of that line (y=mx+b) and a perpendicular line through the
        point in question (y=ax+c). We note that:
        
        - m = (dy/dx)
        - b = edge_y - (dy/dx)(edge_x)
        - a = -(dx/dy) [since it is perpendicular]
        - c = point_y + (dx/dy)(point_x)
        - If mx+b = ax+c, then x=(c-b)/(m-a)
        
        By substituting the first four into the last equation, and simplifying, we find that
        
        x-intersect = [(dx * dy * (point_y - edge_y)) + (dy^2 * edge_x) + (dx^2 * point_x)] / (dy^2 + dx^2)
        
        and y-intersect = m * x-intersect + b
        
        Note that if dx == 0 (vertical line), the x coordinate is simply edge_x. Likewise, if dy == 0, 
        the x coordinate is point_x. These conditions are easy to detect and quickly return an appropriate
        point.
    """
    d_y, d_x, b = line_equation((edge[0], edge[1]))
    if b == None:
        # The line is vertical, need different intercept formula.
        return (edge[0][0], point[1])
    if d_y == 0:
        # The line is horizontal, we can use a faster formula:
        return (point[0], edge[0][1])
    term_1 = d_x * d_y * (point[1] - edge[1][1])
    term_2 = (d_y ** 2) * edge[1][0]
    term_3 = (d_x ** 2) * point[0]
    denom = (d_y ** 2) + (d_x ** 2)
    x_int = (term_1 + term_2 + term_3) / denom
    y_int = (d_y / d_x) * x_int + b
    return (x_int, y_int)
    
def point_sqr_distance(point_a:tuple, point_b:tuple)->float:
    """ Finds the squared distance between two points."""
    return (point_b[1]-point_a[1]) ** 2 + (point_b[0] - point_a[0]) ** 2
    
def between(check:float, boundary_1:float, boundary_2:float)->bool:
    """ Checks if a value is between two boundary values """
    if boundary_1 > boundary_2:
        boundary_1, boundary_2 = boundary_2, boundary_1
    return boundary_1 <= check and check <= boundary_2
        
def near_segment(point:tuple, edge:tuple)->bool:
    """ Checks if a point is within the rectangle with edge as one of the diagonals. """
    return between(point[0], edge[0][0], edge[1][0]) and between(point[1], edge[0][1], edge[1][1])
    
def dot_product(vec_1:tuple, vec_2:tuple)->float:
    """ Dot product of two vectors. """
    return vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1]
    
def magnitude(vector:tuple)->float:
    """ Magnitude of a vector. """
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    
def is_zero_vector(vector:tuple)->bool:
    """ Returns true if the vector is a zero vector, otherwise false. """
    return vector[0] == 0 and vector[1] == 0
    
def vector_cosine_angle(vec_1:tuple, vec_2:tuple)->float:
    """ Cosine of the angle between two vectors. """
    if is_zero_vector(vec_1) or is_zero_vector(vec_2):
        return None
    return dot_product(vec_1, vec_2) / (magnitude(vec_1) * magnitude(vec_2))

def vectorize(point_a:tuple, point_b:tuple)->tuple:
    """ Creates the vector AB from two points. """
    return (point_b[0] - point_a[0], point_b[1] - point_a[1])
    
def line_equation(line:tuple)->tuple:
    """ Determines the three components of a line: d_y, d_x (such that d_y / d_x = slope) and the x-intercept.
    
        Returns
        -------
        tuple:
            Three components: change in y, change in x, and the x-intercept.
            
            If the x-intercept is "None" then dx = 0 and the line is vertical.
    """
    d_y = line[1][1] - line[0][1]
    d_x = line[1][0] - line[0][0]
    b = None
    if not d_x == 0:
        b = line[1][1] - ((d_y / d_x) * line[1][0])
    return d_y, d_x, b
    
def convex_line_segment(point_list:list, desc_y:bool=False, desc_x:bool=False)->list:
    """ Creates a convex line segment between two points.
    
        In the context of polygon creation, desc_y is set to True if we are building
        the top left or top right corner. Desc_X is set to True if we are building the 
        top right or bottom right corner. This impacts the order we accept points and 
        how we interpret the direction of points compared to the existing line.
    
        Params
        ------
        point_list:list
            A list of points
            
        desc_y:bool
            True if points are sorted in descending order of y
            
        desc_x:bool
            True if points are then sorted in descending order of x
    """
    if len(point_list) < 3:
        return point_list
    line = []
    x_extrema = None
    # Since the list is sorted by x second, the last point is actually the
    # first point of the last block of y values in the list (if more than
    # one coordinate has the minimum y value).
    last_point = point_list[-1]
    test_point = -2
    while point_list[test_point][1] == last_point[1]:
        last_point = point_list[test_point]
        test_point -= 1
    for point in point_list:
        # We end when we get to the last point. Points with the same y-value, but
        # more inside x-value won't be on the polygon.
        if point == last_point: 
            break
        # We skip points that are left of the point we have added already.
        if not x_extrema is None:
            if desc_x and x_extrema >= point[0]:
                continue
            elif not desc_x and x_extrema <= point[0]:
                continue
        # If the line is empty, we just add it.
        if not line:
            line.append(point)
            x_extrema = point[0]
            continue
        dir = direction(line[-1], point, last_point)
        if not desc_y == desc_x:
            dir *= -1
        if dir > 0: # if and only if the polygon stays convex by adding this point...
            if len(line) > 1 and collinear(line[-2], line[-1], point):
                # We remove collinear points to match what Graham's scan does.
                del line[-1]
            line.append(point)
            x_extrema = point[0]
    # We end by adding the last point to the list to complete the line.
    line.append(last_point)
    return line


def intersects_right(point:tuple, line:tuple, d_y:float, d_x:float, b:float, include_top:bool=False, inclusive:bool=False)->bool:
    """ Determines if a line segment described by d_y, d_x, and b is right of a point. """
    min_y, max_y = line[0][1], line[1][1]
    if min_y > max_y:
        min_y, max_y = max_y, min_y
    if not between(point[1], min_y, max_y):
        # The point is above or below the line segment.
        return False
    if include_top and point[1] == min_y:
        # We are excluding the bottom and the point is at the bottom.
        return False
    if (not include_top) and point[1] == max_y:
        # We are excluding the top and teh point is at the top.
        return False
    if b == None:
        # Handling vertical lines
        if inclusive and line[0][0] == point[0]:
            return True
        return line[0][0] > point[0]
    x_int = ((point[1]-b) * d_x) / d_y
    if inclusive and x_int == point[0]:
        return True
    return x_int > point[0]
    
def point_on_line(point:tuple, line:tuple, d_y:float, d_x:float, b:float)->bool:
    """ Determines if a point is on the segment. """
    if not near_segment(point, line):
        # Fast fail to handle cases where the point isn't in the bounding rectangle of the line segment.
        return False
    if b == None and point[0] == line[0][0]:
        return True
    return d_y * point[0] == (point[1] - b) * d_x

def crossing_number(point:tuple, edges:list, include_edges:bool=True)->int:
    """ Determines the crossing number of a horizontal positive ray from point with a polygon defined in edges."""
    crossing_number = 0
    for edge in edges:
        d_y, d_x, b = line_equation(edge)
        if include_edges and point_on_line(point, edge, d_y, d_x, b):
            return 1
        if is_horizontal(edge):
            continue
        if intersects_right(point, edge, d_y, d_x, b, positive_slope(edge), include_edges):
            crossing_number += 1
    return crossing_number
    
def pinp_crossing(point:tuple, edges:list, include_edges:bool=True)->bool:
    """ Determines if a point is in a polygon using crossing-number method."""
    return crossing_number(point, edges, include_edges) % 2 == 1
    
    
def pinp_multiple_crossing(points, edges, include_edges = True):
    """ Determines which points are inside a polygon defined by edges. Points are streamed to callers.
        
        This is an adaptation of the above algorithm optimized for many points by only doing the slope 
        and intercept calculations once per edge.
    """
    crossing_number = []
    initialized = False
    for edge in edges:
        d_y, d_x, b = line_equation(edge)
        index = -1
        for point in points:
            index += 1
            if not initialized:
                crossing_number.append([0, False])
            elif crossing_number[index][1]:
                continue
            if include_edges and point_on_line(point, edge, d_y, d_x, b):
                # If the point is on the edge, then we know it is in the polygon.
                crossing_number[index] = [1, True]
                continue
            if is_horizontal(edge):
                # We ignore horizontal edges (unless points are on them, as above).
                continue
            if intersects_right(point, edge, d_y, d_x, b, positive_slope(edge), include_edges):
                crossing_number[index][0] += 1
        initialized = True
    index = 0
    for point in points:
        if crossing_number[index] % 2 == 1:
            yield point
    

def graham_scan(points):
    """ An implementation of Graham's scan using in-place sorting to quickly build a convex hull. """
    if len(points) <= 3:
        return points
    pointList = ExtendedTupleList(points)
    complete_range = pointList.range_within(0, 1)
    first_point = (complete_range[1]["min"][1], complete_range[1]["min"][0])
    newPoints = ExtendedTupleList([])
    for point in pointList:
        square_dist, cosine = line_length_angle((first_point, point))
        new_point = (point[0], point[1], square_dist, cosine)
        newPoints.append(new_point)
    newPoints.double_sort(3, 2, reverse_outside = True, reverse_inside = True)
    hull = ExtendedTupleList([])
    hull.append(first_point)
    hull.append(newPoints[0])
    lastAngle = newPoints[0][3]
    for k in range(1, len(newPoints)):
        if newPoints[k][3] == lastAngle:
            continue
        lastAngle = newPoints[k][3]
        while (len(hull) >= 2 and direction(hull[-2], hull[-1], newPoints[k]) >= 0):
            hull.pop()
        hull.append(newPoints[k])
    real_hull = []
    for point in hull:
        real_hull.append((point[0], point[1]))
    real_hull.append(real_hull[0])
    return real_hull
    
        
def convex_hull(points):
    """ My own home-grown algorithm for building the convex hull.
    
        It works on the following algorithm:
        
        1. Determine the top, bottom, left, and right edges of the polygons.
           These are horizontal or vertical lines only. If there is only one extreme point
           in any particular direction, the "edge" is a point.
        2. Divide the remaining points into five areas:
            1. Quadrant 1 has points above the top of the left edge and right of the top edge.
            2. Quadrant 2 has points above the top of the right edge and left of the top edge.
            3. Quadrant 3 has points below the bottom of the right edge and left of the bottom edge.
            4. Quadrant 4 has points below the bottom of the left edge and right of the bottom edge.
            5. All other point are clearly not on the convex hull.
        3. Build a convex line segment in each quadrant between the two extrema. Generally this goes:
            1. Order the points from the top down (Q1/2) or bottom up (Q3/4), then order them from 
               right to left (Q1/4) or left to right (Q2/3) if they have the same y-coordinate.
            2. Add the first point into the list (the extrema on the top/bottom)
            3. The next point on the list will be on the convex hull if two conditions are true:
               a. The point is left (Q2/3) or right (Q1/4) of the last point added
               b. The point is "outside" (left Q2/3, right Q1/4) of the line segment between the last added point and the last extrema
                  - i.e. it makes a convex turn, not a concave turn
               c. To avoid duplications, we remove an intermediate collinear point as well
        4. We join these four segments together in the appropriate order to make a convex polygon.               
    """
    pointList = ExtendedTupleList(points)
    complete_ranges = pointList.range_within(0, 1)
    # Filters for four quadrants
    filters = [
        ((0, complete_ranges[1]["max"][2], ">="), (1, complete_ranges[0]["max"][2], ">=")), #Q1
        ((0, complete_ranges[1]["max"][1], "<="), (1, complete_ranges[0]["min"][2], ">=")), #Q2
        ((0, complete_ranges[1]["min"][1], "<="), (1, complete_ranges[0]["min"][1], "<=")), #Q3
        ((0, complete_ranges[1]["min"][2], ">="), (1, complete_ranges[0]["max"][1], "<=")) #Q4
    ]
    # Sorting reversals (True means Desc sort, False means Asc sort. Y sort given first)
    sorts = [
        (True, True),
        (True, False),
        (False, False),
        (False, True),
    ]
    hull = ExtendedTupleList([])
    # In CW order of quadrants...
    for index in [0, 3, 2, 1]:
        # Find all the relevant points
        quad_points = ExtendedTupleList([point for point in pointList.filter(filters[index])])
        # Sort them properly
        quad_points.double_sort(1, 0, reverse_outside=sorts[index][0], reverse_inside=sorts[index][1])
        # Build a convex line segment
        line_segment = convex_line_segment(quad_points, sorts[index][0], sorts[index][1])
        # Reverse it, if we need to
        if index % 2 == 1:
            line_segment.reverse()
        # Add all the points in, avoiding repeated points.
        hull.extend(line_segment, avoid_repeats=True)
    return hull
    
def select_longest_edge(hull:list, ignore_left_points:list, min_sqr_length:int=0)->tuple:
    """ Select the longest edge from hull that doesn't start with a point in ignore_left_points, with minimum length.
        
        Returns
        -------
        tuple:
            Two values: first the selected index within hull, and second a tuple in the form (left_point, right_point).
    """
    max_sqr_length = None
    selected = None
    for k in range(0, len(hull) - 1):
        if hull[k] in ignore_left_points:
            continue
        edge_sqr_length = point_sqr_distance(hull[k], hull[k+1])
        if edge_sqr_length < min_sqr_length:
            ignore_left_points.append(hull[k])
            continue
        if max_sqr_length is None or edge_sqr_length > max_sqr_length:
            max_sqr_length = edge_sqr_length
            selected = k
    if not(max_sqr_length is None):
        return selected, (hull[selected], hull[selected + 1])
    else:
        return None, None
    
def segments_intersect(line_1, line_2):
    orient_a = orientation(line_1[0], line_1[1], line_2[0])
    orient_b = orientation(line_1[0], line_1[1], line_2[1])
    if not(orient_a == 0) and orient_a == orient_b:
        return False
    orient_c = orientation(line_2[0], line_2[1], line_1[0])
    orient_d = orientation(line_2[0], line_2[1], line_1[1])
    if not(orient_c == 0) and orient_c == orient_d:
        return False
    if orient_a == 0 and orient_b == 0 and orient_c == 0 and orient_d == 0:
        return near_segment(line_1[0], line_2) or near_segment(line_1[1], line_2)
    return True
    
def extent(point_1, point_2, point_3=None):
    x_min, x_max = point_1[0], point_2[0]
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    y_min, y_max = point_1[1], point_2[1]
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    if not point_3 is None:
        if point_3[0] < x_min:
            x_min = point_3[0]
        elif point_3[0] > x_max:
            x_max = point_3[0]
        if point_3[1] < y_min:
            y_min = point_3[1]
        elif point_3[1] > y_max:
            y_max = point_3[1]
    return (x_min, x_max, y_min, y_max)
    
def rect_overlap(extent_1, extent_2):
    if extent_1[1] < extent_2[0]:
        return False
    if extent_2[1] < extent_1[0]:
        return False
    if extent_1[3] < extent_2[2]:
        return False
    if extent_2[3] < extent_1[2]:
        return False
    return True
    
def segments_intersects_hull(new_edges:list, hull:list, current_edge:tuple)->bool:
    """ Determines if the new edge would intersect the hull at any point."""
    new_edge_extent = extent(new_edges[0][0], new_edges[0][1], new_edges[1][1])
    for k in range(0, len(hull) - 1):
        if new_edges[0][0] == hull[k] or new_edges[0][0] == hull[k+1]:
            continue
        elif new_edges[0][1] == hull[k] or new_edges[0][1] == hull[k+1]:
            continue
        elif new_edges[1][1] == hull[k] or new_edges[1][1] == hull[k+1]:
            continue
        hull_extent = extent(hull[k], hull[k+1])
        if rect_overlap(new_edge_extent, hull_extent):
            test_edge = (hull[k], hull[k+1])
            if segments_intersect(test_edge, new_edges[0]):
                return True
            elif segments_intersect(test_edge, new_edges[1]):
                return True
    return False

def select_candidate_point(edge:tuple, points:list, hull:list, min_cosine:float=-1)->tuple:
    """ Selects the point that meets the conditions to become part of a concave hull.
    
        Conditions:
        - Not on the hull
        - The intersection of a perpendicular line to the edge through the point with the edge exists
        - The concave angle formed will be larger than the minimum angle (using min_cosine of that angle)
        - Would not cause an intersection with the hull
        - Closer to the edge than any other point meeting the above conditions
        
        
        Returns
        -------
        tuple:
            The (x,y) coordinates of the point that can next be used, or None if no such point is left.
    """
    min_sqr_distance = None
    selected = None
    for point in points:
        nearest_point = closest_line_point(point, edge)
        if not near_segment(nearest_point, edge):
            # We ignore points that wouldn't be above or below the edge if rotated horizontal
            continue
        sqr_distance = point_sqr_distance(nearest_point, point)
        if not(min_sqr_distance is None) and min_sqr_distance < sqr_distance:
            # We ignore points that aren't a candidate for minimum distance
            continue
        vector_a = vectorize(edge[0], point)
        vector_b = vectorize(edge[1], point)
        cos_angle = vector_cosine_angle(vector_a, vector_b)
        if cos_angle is None or cos_angle > min_cosine:
            # We ignore points that would make an angle smaller (tighter) than the minimum angle.
            continue
        new_segments = [(edge[0], point), (point, edge[1])]
        if segments_intersects_hull(new_segments, hull, edge):
            # We ignore points that would cause the hull to self-intersect
            continue
        selected = point
        min_sqr_distance = sqr_distance
    return selected
    
def avg_edge_sqr_length(edges:list):
    min_distance = 0
    for k in range(0, len(edges)-1):
        min_distance += point_sqr_distance(edges[k], edges[k+1])
    return min_distance / (len(edges) - 1)
            
    
def concave_hull(hull:list, points:list, max_iterations:int=None, min_length_fraction:float=0, min_angle:float=90)->list:
    """ Creates a concave hull.
        
        The algorithm works like this:
        1. Select the longest edge that is at least min_length in length
        2. Select the closest point to that edge that can make a concave polygon that doesn't have overlapping
           edges and has an concave angle of no less than min_angle.
        3. Replace the edge with two edges using that point as an intermediate point and count this as an iteration.
        4. Alternatively, if there is no suitable point, remove that edge from consideration.
        5. Repeat from 1, until there are no more edges left that can be considered or max_iterations has been reached.
    
    (sqrt(m) * f) ** 2
    """
    tweet.info("Creating concave hull; minimum side length {}% of average, minimum_angle {}".format(min_length_fraction * 100, min_angle))
    test_points = set(points)
    ignore_points = []
    avg_sqr_distance = 0
    for k in range(0, len(hull)-1):
        avg_sqr_distance += point_sqr_distance(hull[k], hull[k+1])
        test_points.remove(hull[k])
    avg_sqr_distance /= len(hull) - 1
    min_sqr_length = avg_sqr_distance * (min_length_fraction ** 2) # since we get sqr_length, we square the fraction
    min_cosine = math.cos(math.radians(min_angle))
    
    while (max_iterations is None or max_iterations > 0) and test_points:
        selection, edge = select_longest_edge(hull, ignore_points, min_sqr_length)
        tweet.info("Considering edge {}; {} points left".format(edge, len(test_points)))
        if selection is None:
            break
        selected_point = select_candidate_point(edge, test_points, hull, min_cosine)
        if selected_point is None:
            # This edge has no more candidate points, so we ignore it in the next pass
            ignore_points.append(edge[0])
            tweet.debug("No candidate point found.")
            continue
        tweet.debug("Found point {}, inserting new edge.".format(selected_point))
        if not max_iterations is None:
            max_iterations -= 1
        # We add the point into the concave hull
        hull.insert(selection + 1, selected_point)
        test_points.remove(selected_point)
    return hull
