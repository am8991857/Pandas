# 10 Minutes to Pandas - Complete Study Guide

## Introduction to Pandas

**Explanation:**
Pandas is a Python library that makes working with data easy and intuitive. Think of it as Excel on steroids - it's designed specifically for data analysis, manipulation, and cleaning. The name "pandas" stands for "Python Data Analysis Library."

**Standard Import:**
```python
import pandas as pd
import numpy as np  # NumPy is used alongside pandas for numerical operations
```

---

## SECTION 1: Basic Data Structures in Pandas
---

### 1.1 Series

**Explanation:**
A Series is like a column in a spreadsheet or a list with labels. It's a 1-dimensional array that holds data of ANY type (integers, strings, floats, etc.). Each element has an associated label called an "index."

**Simple Analogy:**
- Series = A single column in Excel with row labels

**Code Example:**
```python
import pandas as pd
import numpy as np

# Create a Series with default index (0, 1, 2, ...)
s = pd.Series([1, 3, 5, np.nan, 6, 8])
# This creates a Series with 6 values
# np.nan is "Not a Number" - represents missing data
# Index defaults to: 0, 1, 2, 3, 4, 5
# dtype: float64 (because NaN forces float type)

print(s)
# Output:
# 0    1.0
# 1    3.0
# 2    5.0
# 3    NaN
# 4    6.0
# 5    8.0
# dtype: float64
```

**Mini-Example:**
```python
# Create a Series with custom index (labels)
student_grades = pd.Series(
    [95, 87, 92, 88],  # The values (data)
    index=['Alice', 'Bob', 'Charlie', 'Diana']  # Custom labels for each value
)
print(student_grades)
# Output:
# Alice      95
# Bob        87
# Charlie    92
# Diana      88
# dtype: int64
```

**Study Notes:**
- Series = 1D labeled array
- Default index: 0, 1, 2, ... (auto-generated)
- Can have custom index labels
- Supports missing data (NaN)
- dtype shows the data type (int64, float64, object, etc.)

**Quick Tips:**
- Use `len(series)` to get number of elements
- Use `series.index` to see all index labels
- Use `series.values` to get just the data without labels
- Remember: One dimension = Series; Two dimensions = DataFrame

---

### 1.2 DataFrame

**Explanation:**
A DataFrame is like a spreadsheet or a table with rows AND columns. It's a 2-dimensional data structure where each column can have a different data type. Think of it as multiple Series stacked side by side.

**Simple Analogy:**
- DataFrame = An entire Excel spreadsheet with multiple columns

**Code Example:**
```python
# Create a DataFrame with random numbers and date index
dates = pd.date_range("20130101", periods=6)  
# Creates 6 consecutive dates starting from Jan 1, 2013

df = pd.DataFrame(
    np.random.randn(6, 4),  # 6 rows, 4 columns of random numbers
    index=dates,  # Row labels = dates
    columns=list("ABCD")  # Column labels = A, B, C, D
)

print(df)
# Output: A 6x4 table with dates as rows and A,B,C,D as columns
```

**Mini-Example - Create DataFrame from Dictionary:**
```python
# Dictionary way (most intuitive for beginners)
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],  # Column 1: names
    'Age': [25, 30, 35],  # Column 2: ages
    'City': ['NYC', 'LA', 'Chicago']  # Column 3: cities
})

print(df)
# Output:
#      Name  Age     City
# 0   Alice   25      NYC
# 1     Bob   30       LA
# 2  Charlie   35  Chicago
```

**Study Notes:**
- DataFrame = 2D table (rows × columns)
- Can mix different data types in different columns
- Rows have index labels; columns have names
- Best way to work with real datasets
- Can be created from: dictionaries, lists, NumPy arrays, other DataFrames

**Quick Tips:**
- Use `df.shape` to see (rows, columns) dimensions
- Use `df.dtypes` to see data type of each column
- Use `df.info()` for a quick overview of the DataFrame
- Use `df.head()` to preview the first few rows

---

## SECTION 2: Creating Data Structures
---

### 2.1 Creating Series

**Explanation:**
You can create a Series from a list, dictionary, or even a single value. The most common way is to pass a list of values.

**Code Examples:**

```python
# Method 1: From a list (simplest)
s1 = pd.Series([10, 20, 30, 40])
# Creates a Series with default index 0,1,2,3

# Method 2: From a list with custom index
s2 = pd.Series(
    [10, 20, 30, 40],
    index=['a', 'b', 'c', 'd']  # Custom labels
)

# Method 3: From a dictionary
prices = pd.Series({
    'apple': 0.50,
    'banana': 0.30,
    'orange': 0.40
})
# Dictionary keys become the index, values become the data

# Method 4: From a scalar (single value) repeated
s4 = pd.Series(5, index=['x', 'y', 'z'])
# Output: x=5, y=5, z=5 (same value repeated 3 times)
```

**Study Notes:**
- Default index is always 0-based sequence
- Custom index can be any meaningful labels
- Dictionary method: keys→index, values→data
- Missing data is automatically represented as NaN

**Quick Tips:**
- Keep index labels meaningful for easier data retrieval
- Use `.index` attribute to see current index
- Use `.rename_axis()` to rename the index itself

---

### 2.2 Creating DataFrames

**Explanation:**
There are several ways to create a DataFrame, but the most common is from a dictionary of lists or from a 2D array with labels.

**Code Examples:**

```python
# Method 1: From a dictionary (MOST COMMON)
df = pd.DataFrame({
    'A': [1, 2, 3, 4],  # Column A with 4 values
    'B': [10, 20, 30, 40],  # Column B with 4 values
    'C': [100, 200, 300, 400]  # Column C with 4 values
})
# Result: 4 rows × 3 columns, index defaults to 0,1,2,3

# Method 2: From a NumPy array with labels
data = np.random.randn(3, 3)  # 3x3 array of random numbers
df = pd.DataFrame(
    data,
    index=['row1', 'row2', 'row3'],  # Custom row labels
    columns=['col_A', 'col_B', 'col_C']  # Custom column names
)

# Method 3: From a list of lists
df = pd.DataFrame([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], columns=['X', 'Y', 'Z'])
# Result: 3 rows × 3 columns with columns named X, Y, Z

# Method 4: Mixed data types (from dict - most flexible)
df = pd.DataFrame({
    'Numbers': [1, 2, 3],
    'Floats': [1.5, 2.5, 3.5],
    'Strings': ['a', 'b', 'c'],
    'Dates': pd.date_range('2024-01-01', periods=3)
})
# Each column can have a different data type!
```

**Study Notes:**
- Dictionary method is most intuitive and flexible
- All columns must have the same length
- You can mix data types across columns (but not within)
- Use `pd.date_range()` for creating date sequences
- Use `pd.Timestamp()` for single date values

**Quick Tips:**
- When creating from dict, key order is preserved (in Python 3.7+)
- Missing values are automatically filled with NaN
- Use `df.dtypes` to verify data types after creation

---

## SECTION 3: Viewing Data
---

### 3.1 Preview Data

**Explanation:**
When working with large datasets, you don't want to see ALL rows at once. Use `.head()` and `.tail()` to preview the top and bottom rows.

**Code Examples:**

```python
# Assuming we have a DataFrame df with 100 rows

# See the first 5 rows (default)
df.head()  
# Returns the first 5 rows

# See the first 10 rows
df.head(10)
# Specify the number you want

# See the last 5 rows
df.tail()
# Default is 5 rows from the end

# See the last 3 rows
df.tail(3)
# Specify the number from the end

# Quick check: first and last rows
print(df.head(1))  # First row only
print(df.tail(1))  # Last row only
```

**Mini-Example:**
```python
# Create a sample DataFrame
df = pd.DataFrame({
    'Product': ['Apple', 'Banana', 'Orange', 'Mango', 'Grape'],
    'Price': [0.50, 0.30, 0.40, 0.80, 0.60],
    'Stock': [100, 150, 200, 75, 120]
})

print(df.head(2))  # Show first 2 products only
# Output:
#   Product  Price  Stock
# 0   Apple   0.50    100
# 1  Banana   0.30    150
```

**Study Notes:**
- `.head(n)` returns first n rows (default n=5)
- `.tail(n)` returns last n rows (default n=5)
- Useful for verifying data after loading/processing
- Doesn't modify the DataFrame, just displays

**Quick Tips:**
- Always use `.head()` first when loading data to verify it loaded correctly
- Use `.tail()` to find the last few records or check data continuity
- Both methods return a NEW DataFrame (don't modify the original)

---

### 3.2 DataFrame Metadata

**Explanation:**
Before diving into analysis, inspect the DataFrame's structure: dimensions, index, columns, and data types.

**Code Examples:**

```python
# Get the index (row labels)
df.index
# Output: DatetimeIndex or RangeIndex depending on how df was created

# Get the column names
df.columns
# Output: Index(['A', 'B', 'C', 'D'], dtype='object')

# Get the shape (rows, columns)
df.shape
# Output: (6, 4) means 6 rows and 4 columns

# Get data types of all columns
df.dtypes
# Output: Shows each column name and its data type
# Example:
# A    float64
# B    float64
# C    float64
# dtype: object

# Get info about the DataFrame
df.info()
# Output: Shows shape, columns, dtypes, memory usage, non-null counts

# Get the raw NumPy array (without labels)
df.to_numpy()
# Returns a 2D NumPy array with just the data
# Useful when you need to work with raw numerical data

# Statistical summary
df.describe()
# Shows: count, mean, std, min, 25%, 50%, 75%, max for each column
```

**Mini-Example:**
```python
df = pd.DataFrame({
    'Age': [25, 30, 35, 40, np.nan],  # One missing value
    'Score': [85.5, 92.0, 88.5, 95.0, 89.0],
    'Grade': ['A', 'A', 'B', 'A', 'B']
})

print(df.info())
# Output shows: 5 rows, Age has 4 non-null (one NaN), dtypes, etc.

print(df.describe())
# Output: Statistics for Age and Score (not for Grade since it's text)
```

**Study Notes:**
- `.index` = row labels
- `.columns` = column names
- `.shape` = dimensions (rows, columns)
- `.dtypes` = data type of each column
- `.info()` = comprehensive overview with null counts
- `.describe()` = statistical summary (mean, std, min, max, etc.)
- `.to_numpy()` = convert to raw array without labels

**Quick Tips:**
- Use `.info()` immediately after loading data to check for missing values
- Use `.describe()` to quickly spot outliers (compare min/max to mean)
- Remember: `.describe()` only works on numeric columns
- Use `.shape[0]` for row count, `.shape[1]` for column count

---

### 3.3 Transposing Data

**Explanation:**
Transposing flips your DataFrame: rows become columns and columns become rows. Like rotating a table 90 degrees.

**Code Examples:**

```python
# Original DataFrame
#    A    B    C    D
# 0  1    2    3    4
# 1  5    6    7    8

# Transpose it
df_transposed = df.T  # Note: just add .T, it's simple!

# Result:
#    0    1
# A  1    5
# B  2    6
# C  3    7
# D  4    8

# Rows and columns are swapped!
```

**Mini-Example:**
```python
# When transposing is useful:
quarterly_sales = pd.DataFrame({
    'Q1': [1000, 2000, 3000],  # Q1 sales for products
    'Q2': [1200, 2100, 3200],  # Q2 sales for products
    'Q3': [1300, 2200, 3300]   # Q3 sales for products
}, index=['Product_A', 'Product_B', 'Product_C'])

# Original: Products in rows, Quarters in columns
print(quarterly_sales)

# Transposed: Quarters in rows, Products in columns
print(quarterly_sales.T)
# Now you can see all quarters for each product by row
```

**Study Notes:**
- Transpose using `.T` attribute (no parentheses!)
- Useful for changing perspective: from wide to long or vice versa
- Row/column names swap positions

**Quick Tips:**
- Transpose doesn't modify original (creates new view)
- Use when you need to work with data in a different orientation
- Useful before exporting data in a different format

---

## SECTION 4: Sorting Data
---

### 4.1 Sorting by Index

**Explanation:**
Sort a DataFrame by its row labels (index). Useful when index has meaning (like dates or names).

**Code Examples:**

```python
# Sort by index in ascending order (default)
df_sorted = df.sort_index()
# Returns a new DataFrame sorted by index A→Z or oldest→newest

# Sort by index in descending order
df_sorted = df.sort_index(ascending=False)
# Reverse order: Z→A or newest→oldest

# Sort by columns (axis=1)
df_sorted = df.sort_index(axis=1, ascending=False)
# Sorts COLUMN names in reverse order: D, C, B, A instead of A, B, C, D

# Sort only specific columns
df_sorted = df.sort_index(level=0)  # For MultiIndex DataFrames
```

**Mini-Example:**
```python
# Create DataFrame with non-sorted index
df = pd.DataFrame({
    'Value': [100, 200, 300, 400]
}, index=['D', 'B', 'C', 'A'])

print(df.sort_index())
# Output: Rows in order A, B, C, D (by index)
#    Value
# A    400
# B    200
# C    300
# D    100
```

**Study Notes:**
- `.sort_index()` sorts by row labels
- Use `ascending=False` for reverse order
- Use `axis=1` to sort columns instead of rows
- Returns a new DataFrame (doesn't modify original)

**Quick Tips:**
- Default ascending=True (A-Z, oldest-newest)
- For dates, ascending=True gives oldest first
- Useful for organizing data chronologically or alphabetically

---

### 4.2 Sorting by Values

**Explanation:**
Sort a DataFrame by the actual data in one or more columns. For example, sort sales from highest to lowest.

**Code Examples:**

```python
# Sort by column B in ascending order
df_sorted = df.sort_values(by='B')
# Rows are rearranged so column B values go from smallest to largest

# Sort by column B in descending order
df_sorted = df.sort_values(by='B', ascending=False)
# Highest values first

# Sort by multiple columns (primary, then secondary sort)
df_sorted = df.sort_values(by=['A', 'B'])
# First sort by A, then by B for rows where A values are equal

# Sort by multiple columns with different orders
df_sorted = df.sort_values(
    by=['A', 'B'],
    ascending=[True, False]  # A ascending, B descending
)

# Keep/drop original index after sorting
df_sorted = df.sort_values(by='B', ignore_index=True)
# Creates new index 0,1,2,... instead of keeping original
```

**Mini-Example:**
```python
# Student grades DataFrame
grades_df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Math': [95, 88, 92, 87],
    'English': [85, 90, 88, 92]
})

# Sort by Math score (highest first)
print(grades_df.sort_values(by='Math', ascending=False))
# Output:
#      Name  Math  English
# 0   Alice    95       85
# 2  Charlie   92       88
# 1     Bob    88       90
# 3    Diana   87       92

# Sort by Math, then by English (if Math equal)
print(grades_df.sort_values(by=['Math', 'English']))
```

**Study Notes:**
- `.sort_values(by='column_name')` sorts by specific column
- `ascending=False` for descending order (highest first)
- Can sort by multiple columns simultaneously
- Default: returns new DataFrame, keeps original

**Quick Tips:**
- `by=` parameter expects column name(s)
- Use `ascending=[True, False]` for mixed sort orders
- Use `ignore_index=True` to reset index after sorting
- Remember: `sort_values()` sorts by DATA, `sort_index()` sorts by LABELS

---

## SECTION 5: Selecting Data (Indexing)
---

### 5.1 Column Selection (Getitem [])

**Explanation:**
Get data from your DataFrame by selecting specific columns. This is one of the most common operations.

**Code Examples:**

```python
# SELECT ONE COLUMN - Returns a Series
column_A = df['A']
# Gets all values from column A as a 1D Series
# Same as: df.A (if column name is simple without spaces/special chars)

# Alternative syntax (only works if column name has no spaces)
column_A = df.A  
# Less explicit but shorter

# SELECT MULTIPLE COLUMNS - Returns a DataFrame
subset_df = df[['B', 'A']]  
# Note the DOUBLE brackets: outer for getitem, inner for list
# Returns a new DataFrame with just columns B and A
# Also reorders columns: B first, then A

# SELECT ROWS by position slice
first_three = df[0:3]  
# Gets rows at positions 0, 1, 2 (NOT row 3)
# Remember: Python slicing is exclusive of the end point

# SELECT ROWS by date (if index is DatetimeIndex)
date_range = df['20130102':'20130104']  
# Gets rows with dates between Jan 2 and Jan 4 (both INCLUSIVE!)
# Note: Date slicing INCLUDES both endpoints (different from integer slicing!)
```

**Mini-Example:**
```python
sales_df = pd.DataFrame({
    'Product': ['Apple', 'Banana', 'Orange'],
    'Q1': [100, 150, 200],
    'Q2': [120, 160, 210],
    'Q3': [110, 155, 205]
})

# Get one column
q1_sales = sales_df['Q1']
# Returns: Series with [100, 150, 200]

# Get multiple columns
quarterly = sales_df[['Q1', 'Q2', 'Q3']]
# Returns: DataFrame with just those columns

# Reorder columns
reordered = sales_df[['Q3', 'Q1', 'Q2']]
# Columns appear in this new order: Q3, Q1, Q2
```

**Study Notes:**
- `df['column_name']` returns a Series (1D)
- `df[['col1', 'col2']]` returns a DataFrame (2D)
- Double brackets needed when selecting multiple columns
- Can also reorder columns by listing in desired order
- Integer slicing excludes the end (0:3 = 0,1,2)
- Date slicing INCLUDES the end date

**Quick Tips:**
- Single column → Series; Multiple columns → DataFrame
- Remember the double brackets: `[['col1', 'col2']]`
- Use this method for column selection, use `.loc[]` for row selection
- Can rearrange columns just by changing the order in the list

---

### 5.2 Selection by Label (.loc[])

**Explanation:**
Use `.loc[]` to select rows and columns by their LABELS (names/index values). Most intuitive for labeled data.

**Code Examples:**

```python
# SELECT one row by its index label
row = df.loc[dates[0]]  
# Gets the row with index label = dates[0]
# Returns a Series with all column values for that row

# SELECT all rows (:) but only specific columns
subset = df.loc[:, ['A', 'B']]
# : means "all rows"
# ['A', 'B'] selects those columns
# Returns a DataFrame with all rows, columns A and B only

# SELECT by date range (BOTH ENDPOINTS INCLUSIVE)
subset = df.loc['20130102':'20130104', ['A', 'B']]
# Gets rows from Jan 2 to Jan 4 (both included!)
# And columns A, B only
# Returns a DataFrame

# SELECT a single value (scalar)
value = df.loc[dates[0], 'A']
# Gets ONE specific value: row dates[0], column A
# Much faster than getting entire row then column

# SELECT by condition (boolean indexing)
filtered = df.loc[df['A'] > 0]
# Gets all rows where column A value > 0
# Returns a DataFrame with those rows
```

**Mini-Example:**
```python
employees = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Salary': [50000, 60000, 70000]
}, index=['E001', 'E002', 'E003'])

# Get row with label 'E002'
print(employees.loc['E002'])
# Output: Name=Bob, Age=30, Salary=60000 (Series)

# Get specific columns for all employees
print(employees.loc[:, ['Name', 'Salary']])
# All rows, but only Name and Salary columns

# Get one specific value
alice_salary = employees.loc['E001', 'Salary']
# Output: 50000 (scalar value)
```

**Study Notes:**
- `.loc[]` uses LABELS (index/column names)
- Format: `.loc[row_label, column_label]`
- Use `:` to mean "all rows" or "all columns"
- Date slicing with .loc[] is INCLUSIVE on both ends
- Returns: Series (single row), DataFrame (multiple rows), or scalar (single cell)

**Quick Tips:**
- `.loc[dates[0]]` gets entire row
- `.loc[:, 'column_name']` gets entire column
- `.loc[start:end]` includes both endpoints (date slicing)
- Use for labeled/index-based selection (most intuitive)

---

### 5.3 Selection by Position (.iloc[])

**Explanation:**
Use `.iloc[]` to select rows and columns by their POSITION (integer location), like array indexing. Similar to NumPy.

**Code Examples:**

```python
# SELECT one row by position
row = df.iloc[3]
# Gets row at position 3 (4th row, since counting starts at 0)
# Returns a Series

# SELECT rows by position slice
subset = df.iloc[3:5]
# Gets rows at positions 3 and 4 (NOT 5)
# Remember: end position is EXCLUSIVE

# SELECT rows AND columns by position
subset = df.iloc[3:5, 0:2]
# Rows 3-4, Columns 0-1 (A and B)
# Returns a DataFrame

# SELECT specific rows and columns by list
subset = df.iloc[[1, 2, 4], [0, 2]]
# Rows at positions 1, 2, 4
# Columns at positions 0, 2 (columns A and C)
# Returns a DataFrame with these specific rows/columns

# SELECT one specific value (scalar)
value = df.iloc[1, 1]
# Row position 1, Column position 1 (second row, second column)
# Returns a scalar value

# SELECT entire row by position
row = df.iloc[1, :]
# Row 1 (second row), all columns

# SELECT entire column by position
col = df.iloc[:, 0]
# All rows, Column 0 (first column)
```

**Mini-Example:**
```python
# DataFrame with 5 rows, 3 columns
df = pd.DataFrame({
    'A': [10, 20, 30, 40, 50],
    'B': [100, 200, 300, 400, 500],
    'C': [1000, 2000, 3000, 4000, 5000]
})

print(df.iloc[2])  # Row at position 2 (3rd row)
# Output: A=30, B=300, C=3000

print(df.iloc[1:4, 0:2])  # Rows 1-3, Columns A-B
# Output: 3x2 DataFrame

print(df.iloc[[0, 2, 4], [1, 2]])  # Rows 0,2,4 and columns B,C
# Output: 3x2 DataFrame
```

**Study Notes:**
- `.iloc[]` uses POSITIONS (integer indices)
- Format: `.iloc[row_position, column_position]`
- Slicing is EXCLUSIVE of the end position (like Python)
- Use integers for positions
- Can use lists to select non-consecutive rows/columns

**Quick Tips:**
- `.iloc[1]` is faster than `.loc[]` when using positions
- Remember: end position is exclusive (different from `.loc[]`)
- Use for positional selection (useful for first/last N rows)
- Combine with conditions: `df.iloc[0:10]` for first 10 rows

---

### 5.4 Boolean Indexing

**Explanation:**
Filter rows based on conditions. For example, get only rows where a column value meets a criteria.

**Code Examples:**

```python
# SELECT rows where column A > 0
filtered = df[df['A'] > 0]
# df['A'] > 0 creates a boolean Series (True/False for each row)
# df[boolean_series] returns only rows where True
# Returns a DataFrame

# SELECT rows with column B = 10
filtered = df[df['B'] == 10]
# == for equality, = is assignment (don't mix them up!)

# MULTIPLE conditions with & (AND)
filtered = df[(df['A'] > 0) & (df['B'] < 100)]
# Rows where BOTH conditions are true
# Need parentheses around each condition!
# & means AND (not 'and' keyword)

# MULTIPLE conditions with | (OR)
filtered = df[(df['A'] > 50) | (df['B'] == 20)]
# Rows where AT LEAST ONE condition is true
# | means OR (not 'or' keyword)

# Invert condition with ~
filtered = df[~(df['A'] > 0)]
# Get rows where A is NOT > 0
# ~ is the NOT operator

# USE isin() for multiple values
filtered = df[df['Status'].isin(['Active', 'Pending'])]
# Get rows where Status is either 'Active' OR 'Pending'
# Cleaner than: (df['Status'] == 'Active') | (df['Status'] == 'Pending')

# SELECT values where condition is met, rest = NaN
result = df[df > 0]
# For each cell: keep if > 0, otherwise set to NaN
# Useful for filtering while keeping structure
```

**Mini-Example:**
```python
students = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Math': [95, 88, 92, 87],
    'English': [85, 90, 88, 92]
})

# Get students with Math score >= 90
print(students[students['Math'] >= 90])
# Output: Alice and Charlie rows

# Get students who scored 90+ in either subject
print(students[(students['Math'] >= 90) | (students['English'] >= 90)])
# Output: Alice, Bob, Charlie (Diana excluded)

# Use isin() for cleaner multiple value filtering
names_to_find = ['Alice', 'Diana']
print(students[students['Name'].isin(names_to_find)])
# Output: Alice and Diana rows
```

**Study Notes:**
- Boolean indexing creates a True/False filter
- Operators: `>`, `<`, `==`, `!=`, `>=`, `<=`
- Multiple conditions: use `&` (AND), `|` (OR), `~` (NOT)
- MUST use parentheses around each condition
- Use `.isin()` for checking membership in a list
- Can also use `.str.contains()` for string matching

**Quick Tips:**
- Common mistake: using `and` instead of `&` - this WILL break!
- Remember: parentheses are REQUIRED around each condition
- Use `.isin()` when checking multiple specific values
- Boolean indexing returns a new DataFrame (doesn't modify original)
- Use `df[df > 0]` to keep structure but replace non-matching values with NaN

---

## SECTION 6: Setting Data (Modification)
---

### 6.1 Setting Values

**Explanation:**
Modify DataFrame values: add new columns, update existing values, or fill data.

**Code Examples:**

```python
# ADD a new column to DataFrame
df['new_column'] = [1, 2, 3, 4, 5]
# New column automatically aligns by index

# ADD a Series as a column
s = pd.Series([1, 2, 3, 4, 5], index=df.index)
df['F'] = s
# Series index aligns with DataFrame index automatically

# SET a value by label
df.at[dates[0], 'A'] = 0
# Set row dates[0], column A to 0
# .at[] is fastest for single value assignment

# SET a value by position
df.iat[0, 1] = 0
# Set row 0, column 1 to 0
# .iat[] is fastest for positional assignment

# SET entire column
df.loc[:, 'D'] = np.array([5] * len(df))
# Set all values in column D to 5

# SET with condition (where operation)
df2 = df.copy()  # Create a copy first
df2[df2 > 0] = -df2[df2 > 0]
# Where values > 0, replace with negative of that value

# ADD column with condition
df['category'] = df['A'] > 0  # True/False values
# Or use np.where()
df['category'] = np.where(df['A'] > 0, 'Positive', 'Negative')
# If A > 0: 'Positive', else: 'Negative'
```

**Mini-Example:**
```python
# Create a simple DataFrame
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Score': [85, 92, 78]
})

# Add a new column
df['Grade'] = ['B', 'A', 'C']

# Update a single value
df.at[0, 'Score'] = 90  # Change Alice's score to 90

# Add calculated column
df['Pass'] = df['Score'] >= 80  # True if score >= 80

print(df)
# Output:
#      Name  Score Grade  Pass
# 0   Alice     90     B  True
# 1     Bob     92     A  True
# 2  Charlie     78     C False
```

**Study Notes:**
- `df['column'] = values` adds or updates a column
- `.at[row, col]` sets single value by label (fast!)
- `.iat[row_pos, col_pos]` sets single value by position (fast!)
- `.loc[:, 'col']` can set entire column
- Pandas automatically aligns by index when adding Series
- Use `df.copy()` before modifying to keep original safe

**Quick Tips:**
- Use `.at[]` or `.iat[]` for single cell updates (faster than .loc[])
- Always use `.copy()` when creating modified versions to avoid SettingWithCopyWarning
- Adding columns is fast; adding rows requires copy (slow)
- Use `np.where()` for complex conditional assignments

---

## SECTION 7: Handling Missing Data
---

### 7.1 Understanding Missing Data

**Explanation:**
Real data always has missing values (NaN = Not a Number). You need to handle them: remove, fill, or ignore.

**Code Examples:**

```python
# CHECK for missing values
mask = pd.isna(df)
# Returns boolean DataFrame: True where value is NaN

# COUNT missing values
df.isnull().sum()  # Count missing per column
# Output: Number of NaN values in each column

# DROP rows with ANY missing values
df_clean = df.dropna(how='any')
# Removes entire row if it has ANY NaN

# DROP rows only if ALL values are NaN
df_clean = df.dropna(how='all')
# Removes row only if EVERY value is NaN

# DROP rows where specific column is NaN
df_clean = df.dropna(subset=['A', 'B'])
# Remove rows if column A or B has NaN

# FILL missing values with a constant
df_filled = df.fillna(0)
# Replace all NaN with 0

# FILL missing values with a method
df_filled = df.fillna(method='ffill')  # Forward fill
# Use previous value to fill NaN

df_filled = df.fillna(method='bfill')  # Backward fill
# Use next value to fill NaN

# FILL missing with different values per column
df_filled = df.fillna({'A': 0, 'B': 99, 'C': 'Unknown'})
# Fill each column with appropriate value

# INTERPOLATE (for time series data)
df_interpolated = df.interpolate()
# Estimate missing values based on neighboring values
```

**Mini-Example:**
```python
# DataFrame with missing values
df = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=5),
    'Temperature': [20, np.nan, 22, 21, np.nan],
    'Humidity': [65, 70, np.nan, 72, 73]
})

print(df.isnull())
# Shows where NaN values are

print(df.isnull().sum())
# Output: Temperature has 2 NaN, Humidity has 1 NaN

# Fill with forward fill (use previous value)
print(df.fillna(method='ffill'))
# Temperature becomes [20, 20, 22, 21, 21]

# Fill with specific values
print(df.fillna({'Temperature': 21, 'Humidity': 70}))
# Fill Temperature NaN with 21, Humidity NaN with 70
```

**Study Notes:**
- `np.nan` and `pd.NA` represent missing data
- `pd.isna()` checks if value is NaN (boolean)
- `.isnull()` is alias for `.isna()`
- `.dropna()` removes rows with NaN
- `.fillna()` replaces NaN with values
- `.interpolate()` estimates missing values

**Quick Tips:**
- Check `.isnull().sum()` immediately after loading data
- Decide early: drop or fill? Dropping loses data, filling might introduce bias
- Use `how='any'` vs `how='all'` carefully (any = strictest)
- Forward fill works for time series; doesn't work for random missing data

---

## SECTION 8: Operations
---

### 8.1 Basic Statistics

**Explanation:**
Quickly calculate statistical summaries of your data: mean, sum, count, etc.

**Code Examples:**

```python
# MEAN (average) of each column
means = df.mean()
# Output: Average value for each column

# MEAN of each row
row_means = df.mean(axis=1)
# axis=0 (default) = column-wise
# axis=1 = row-wise

# SUM of each column
sums = df.sum()
# Output: Total for each column

# COUNT of non-NaN values
counts = df.count()
# Output: Number of non-missing values per column

# STANDARD DEVIATION
std = df.std()
# Output: Spread/variability of each column

# MIN and MAX
minimums = df.min()
maximums = df.max()

# QUARTILES (25%, 50%, 75%)
quartiles = df.quantile([0.25, 0.5, 0.75])

# DESCRIBE - Summary statistics
summary = df.describe()
# Output: count, mean, std, min, 25%, 50%, 75%, max

# MEDIAN (50th percentile)
medians = df.median()

# Custom operation: sum all values
total = df.values.sum()  # All values in entire DataFrame
```

**Mini-Example:**
```python
# Sales data
sales = pd.DataFrame({
    'Product': ['A', 'B', 'C', 'D'],
    'Q1': [100, 150, 200, 120],
    'Q2': [110, 160, 210, 130],
    'Q3': [120, 170, 220, 140]
})

print(sales.mean())  # Average sales per quarter
# Output: Q1=142.5, Q2=152.5, Q3=162.5

print(sales.mean(axis=1))  # Average sales per product
# Output: A=110, B=160, C=210, D=130

print(sales.describe())
# Shows: count, mean, std, min, 25%, 50%, 75%, max
```

**Study Notes:**
- `axis=0` performs operation down columns (default)
- `axis=1` performs operation across rows
- Most methods ignore NaN values automatically
- `.describe()` only shows numeric columns
- Common methods: `.sum()`, `.mean()`, `.median()`, `.std()`, `.min()`, `.max()`

**Quick Tips:**
- Use `.describe()` to quickly spot outliers
- Compare mean vs median to detect skewed data
- Use `axis=1` when you want row-wise calculations
- All operations ignore NaN by default (use `skipna=False` to include)

---

### 8.2 Apply & Transform

**Explanation:**
Apply custom functions to your data. `.agg()` reduces data (returns fewer rows), `.transform()` broadcasts (keeps same shape).

**Code Examples:**

```python
# APPLY a function to each column (reduction)
result = df.agg(lambda x: np.mean(x) * 5.6)
# Multiply mean by 5.6 for each column
# Returns one value per column (reduces data)

# APPLY multiple functions
result = df.agg(['mean', 'std', 'min', 'max'])
# Returns multiple statistics per column

# APPLY different functions per column
result = df.agg({'A': 'mean', 'B': 'sum', 'C': 'max'})
# Different function for each column

# TRANSFORM - Apply function element-wise (same shape)
result = df.transform(lambda x: x * 101.2)
# Multiply each value by 101.2
# Returns same shape DataFrame (not reduced)

# APPLY row-wise
result = df.apply(lambda row: row['A'] + row['B'], axis=1)
# Add columns A+B for each row
# Returns Series with one value per row

# APPLY to Series
result = df['A'].apply(lambda x: x ** 2)
# Square each value in column A
# Returns Series
```

**Mini-Example:**
```python
scores = pd.DataFrame({
    'Math': [85, 92, 78, 95],
    'English': [80, 88, 85, 90],
    'Science': [90, 87, 82, 93]
})

# Scale all scores (multiply by 1.1)
scaled = scores.transform(lambda x: x * 1.1)
# Returns same shape, all values scaled

# Get average of all subjects per student
avg = scores.apply(lambda row: row.mean(), axis=1)
# Returns: [85, 89, 81.67, 92.67] (one average per student)

# Get statistics per subject
stats = scores.agg(['mean', 'std', 'min', 'max'])
# Returns table with 4 rows (statistics) × 3 columns
```

**Study Notes:**
- `.agg()` or `.aggregate()` - reduces data (fewer rows)
- `.transform()` - keeps same shape (broadcasts)
- `.apply()` - flexible, works on rows or columns
- Lambda functions (`lambda x:`) are common for quick operations
- Use `axis=0` for columns (default), `axis=1` for rows

**Quick Tips:**
- Use `.agg()` when you want summary statistics (one value per column)
- Use `.transform()` when you want scaled/normalized data (same shape)
- Use `.apply()` when you need more complex custom logic
- Remember: `.agg()` reduces, `.transform()` maintains shape

---

## SECTION 9: Value Counts
---

### 9.1 Counting Occurrences

**Explanation:**
Count how many times each unique value appears in a column. Useful for finding the most common values.

**Code Examples:**

```python
# COUNT occurrences of each value
s = pd.Series([1, 2, 1, 2, 2, 3, 1, 1, 3, 3])

counts = s.value_counts()
# Output:
# 1    4  (value 1 appears 4 times)
# 2    3  (value 2 appears 3 times)
# 3    3  (value 3 appears 3 times)

# GET counts sorted ascending (least common first)
counts = s.value_counts(ascending=True)
# Output: Most common last

# GET all values including NaN
counts = s.value_counts(dropna=False)
# Includes NaN in the counts

# GET proportions instead of counts
proportions = s.value_counts(normalize=True)
# Output: 1 appears in 40%, 2 in 30%, 3 in 30%

# GET top N values
top_3 = s.value_counts().head(3)
# Returns counts for 3 most common values
```

**Mini-Example:**
```python
# Survey responses
survey = pd.Series([
    'Yes', 'No', 'Yes', 'Maybe', 'Yes', 'No', 'No', 'Yes', 'Maybe'
])

print(survey.value_counts())
# Output:
# Yes      4
# No       3
# Maybe    2

print(survey.value_counts(normalize=True))
# Output:
# Yes      0.44  (44%)
# No       0.33  (33%)
# Maybe    0.22  (22%)
```

**Study Notes:**
- `.value_counts()` returns sorted counts (most common first)
- Useful for finding mode (most frequent value)
- Can show proportions with `normalize=True`
- Can include NaN counts with `dropna=False`
- Returns Series with value as index, count as value

**Quick Tips:**
- Perfect for exploratory data analysis (EDA)
- Use `normalize=True` to see percentages instead of raw counts
- Chain with `.head(n)` to get top N values
- Use on categorical columns or Series before visualization

---

## SECTION 10: String Methods
---

### 10.1 String Operations

**Explanation:**
Perform text operations on Series containing strings. Access via `.str` accessor.

**Code Examples:**

```python
# CREATE a Series with strings
s = pd.Series(["A", "B", "C", "Aaba", "Baca", np.nan, "CABA", "dog", "cat"])

# CONVERT to lowercase
lower = s.str.lower()
# Output: a, b, c, aaba, baca, nan, caba, dog, cat

# CONVERT to uppercase
upper = s.str.upper()
# Output: A, B, C, AABA, BACA, NAN, CABA, DOG, CAT

# GET length of each string
lengths = s.str.len()
# Output: 1, 1, 1, 4, 4, nan, 4, 3, 3

# CHECK if string contains substring
contains_a = s.str.contains('a')
# Output: True where string contains 'a'

# REPLACE substring
replaced = s.str.replace('a', 'X')
# Replace all 'a' with 'X'

# SPLIT strings
split = s.str.split('')
# Split each string into characters

# STRIP whitespace
stripped = s.str.strip()
# Remove leading/trailing spaces

# ACCESS substring
substring = s.str[0:2]
# Get first 2 characters of each string
```

**Mini-Example:**
```python
emails = pd.Series([
    'alice@example.com',
    'bob@test.com',
    'charlie@example.com',
    'diana@test.com'
])

# Extract domain
domain = emails.str.split('@').str[1]
# Output: example.com, test.com, example.com, test.com

# Check if Gmail
is_gmail = emails.str.contains('gmail')
# Output: All False (no Gmail addresses)

# Convert to uppercase
upper_emails = emails.str.upper()
# Output: All uppercase
```

**Study Notes:**
- Access string methods via `.str` accessor
- Works only on string/object dtype columns
- Ignores NaN values automatically
- Returns Series (same shape as input)
- Common methods: `.lower()`, `.upper()`, `.len()`, `.contains()`, `.replace()`, `.split()`

**Quick Tips:**
- Use `.str.contains()` for filtering text data
- Use `.str.split()` to extract parts (like domain from email)
- Chain string methods: `s.str.lower().str.strip().str.contains('a')`
- NaN values stay as NaN (not converted to string 'nan')

---

## SECTION 11: Combining Data
---

### 11.1 Concatenation

**Explanation:**
Combine multiple DataFrames by stacking them on top of each other (row-wise) or side-by-side (column-wise).

**Code Examples:**

```python
# CREATE sample DataFrames
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
df3 = pd.DataFrame({'A': [9, 10], 'B': [11, 12]})

# CONCATENATE row-wise (stack on top)
result = pd.concat([df1, df2, df3])
# Result: df1 rows, then df2 rows, then df3 rows
# Indices are preserved: 0,1,0,1,0,1

# RESET index after concatenating
result = pd.concat([df1, df2, df3], ignore_index=True)
# New index: 0,1,2,3,4,5 (sequential)

# CONCATENATE column-wise (side-by-side)
result = pd.concat([df1, df2], axis=1)
# Columns from df1, then columns from df2

# CONCATENATE with different columns
df_a = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df_b = pd.DataFrame({'C': [5, 6], 'D': [7, 8]})
result = pd.concat([df_a, df_b], axis=1)
# Result: Columns A, B, C, D (all aligned by index)
```

**Mini-Example:**
```python
# Sales data for different quarters
q1 = pd.DataFrame({
    'Product': ['A', 'B', 'C'],
    'Sales': [100, 150, 200]
})

q2 = pd.DataFrame({
    'Product': ['A', 'B', 'C'],
    'Sales': [110, 160, 210]
})

# Stack quarters (get all quarters in one DataFrame)
yearly = pd.concat([q1, q2], ignore_index=True)
# Result: 6 rows total (q1 data, then q2 data)

# Alternatively, add quarter label
q1['Quarter'] = 'Q1'
q2['Quarter'] = 'Q2'
combined = pd.concat([q1, q2], ignore_index=True)
# Now can distinguish which quarter each row belongs to
```

**Study Notes:**
- `pd.concat()` combines multiple DataFrames
- `axis=0` (default) stacks row-wise (vertically)
- `axis=1` stacks column-wise (horizontally)
- `ignore_index=True` creates new sequential index
- Preserves index values by default
- Missing columns filled with NaN when aligning

**Quick Tips:**
- Use `ignore_index=True` when concatenating multiple small DataFrames
- Use `axis=1` when joining DataFrames with same index but different columns
- Remember to reset index when combining if you want sequential numbering
- Check column names match before concatenating!

---

### 11.2 Merging (SQL Join)

**Explanation:**
Combine DataFrames based on common columns or indices, like a SQL JOIN operation.

**Code Examples:**

```python
# CREATE sample DataFrames with common column
left = pd.DataFrame({
    'key': ['foo', 'bar'],
    'lval': [1, 2]
})

right = pd.DataFrame({
    'key': ['foo', 'bar'],
    'rval': [4, 5]
})

# MERGE on common column
result = pd.merge(left, right, on='key')
# Combines rows where 'key' values match
# Result: 2 rows with columns: key, lval, rval

# MERGE on index
result = pd.merge(left, right, left_index=True, right_index=True)
# Combines based on row index, not column

# DIFFERENT merge types
# Inner (default): Only matching keys
result = pd.merge(left, right, on='key', how='inner')

# Left: All rows from left, matching rows from right
result = pd.merge(left, right, on='key', how='left')

# Right: All rows from right, matching rows from left
result = pd.merge(left, right, on='key', how='right')

# Outer: All rows from both (full outer join)
result = pd.merge(left, right, on='key', how='outer')

# MERGE with different column names
df1 = pd.DataFrame({'id': [1, 2, 3], 'value_x': [10, 20, 30]})
df2 = pd.DataFrame({'id': [1, 2, 3], 'value_y': [100, 200, 300]})
result = pd.merge(df1, df2, on='id')
```

**Mini-Example:**
```python
# Customers and their orders
customers = pd.DataFrame({
    'CustomerID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana']
})

orders = pd.DataFrame({
    'OrderID': [101, 102, 103],
    'CustomerID': [1, 2, 1],
    'Amount': [50, 100, 75]
})

# Join customers with their orders
result = pd.merge(customers, orders, on='CustomerID', how='left')
# Result: All customers, with matching orders (Diana has no orders = NaN)

# Output:
#   CustomerID      Name  OrderID  Amount
# 0           1     Alice      101      50
# 1           1     Alice      103      75
# 2           2       Bob      102     100
# 3           4     Diana      NaN      NaN
```

**Study Notes:**
- `pd.merge()` is like SQL JOIN
- Default `how='inner'` - only matching rows
- Use `how='left'` to keep all from left DataFrame
- Use `on='column_name'` to join on specific column
- Can also use `left_on` and `right_on` for different column names
- Missing values become NaN in the result

**Quick Tips:**
- Always specify which column(s) to join on
- Use `how='left'` most often to keep all your original data
- Check for duplicate rows after merge (could create many rows!)
- Use `left_on='col1', right_on='col2'` when column names differ

---

## SECTION 12: Grouping Data
---

### 12.1 Group By Operations

**Explanation:**
Split data into groups, apply operations to each group separately, then combine results. Like "split-apply-combine."

**Code Examples:**

```python
# CREATE sample data with groups
df = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Value1': [10, 20, 30, 40, 50, 60],
    'Value2': [100, 200, 300, 400, 500, 600]
})

# GROUP BY single column and sum
grouped_sum = df.groupby('Category')[['Value1', 'Value2']].sum()
# Groups A and B separately, sums their values
# Result:
#           Value1  Value2
# Category
# A             90    900
# B            120   1260

# GROUP BY and get mean
grouped_mean = df.groupby('Category').mean()
# Average for each group

# GROUP BY multiple columns
grouped = df.groupby(['Category', 'Subcategory']).sum()
# Creates hierarchical groups (MultiIndex)

# GROUP BY and count
grouped_count = df.groupby('Category').size()
# Output: A has 3 rows, B has 3 rows

# GROUP BY and get multiple stats
grouped_stats = df.groupby('Category').agg({
    'Value1': 'sum',
    'Value2': 'mean'
})
# Sum Value1, mean of Value2, for each group

# GROUP BY and apply custom function
grouped_custom = df.groupby('Category').apply(lambda x: x.max() - x.min())
# Get range (max - min) for each group

# ITERATE through groups
for name, group in df.groupby('Category'):
    print(f"Group {name}:")
    print(group)
    # Process each group separately
```

**Mini-Example:**
```python
# Sales by region and product
sales = pd.DataFrame({
    'Region': ['North', 'North', 'South', 'South', 'East', 'East'],
    'Product': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Amount': [100, 150, 200, 120, 180, 160]
})

# Total sales per region
region_total = sales.groupby('Region')['Amount'].sum()
# Output:
# Region
# North    250
# South    320
# East     340

# Total sales per product
product_total = sales.groupby('Product')['Amount'].sum()
# Output:
# Product
# A    480
# B    430

# Average sales per region
region_avg = sales.groupby('Region')['Amount'].mean()
```

**Study Notes:**
- `.groupby('column')` creates groups based on unique values
- Can group by single column or multiple columns
- Most common operations: `.sum()`, `.mean()`, `.count()`, `.size()`, `.max()`, `.min()`
- `.agg()` allows different functions per column
- Results automatically indexed by group key
- Use `.apply()` for custom operations per group

**Quick Tips:**
- Use `observed=False` in groupby to show empty groups for categorical data
- Chain groupby operations: `df.groupby('col1')['col2'].sum()`
- Use `.get_group('value')` to get one specific group
- Remember: groupby doesn't modify original DataFrame
- Use `.transform()` within groupby to keep original shape with group results

---

## SECTION 13: Reshaping Data
---

### 13.1 Stack and Unstack

**Explanation:**
Stack and unstack reshape DataFrames by pivoting columns to rows and vice versa.

**Code Examples:**

```python
# STACK - Move column level down to row index (wide to long)
stacked = df.stack()
# Column headers become part of the row index
# Creates MultiIndex

# UNSTACK - Move row index level to columns (long to wide)
unstacked = stacked.unstack()
# Gets back original structure

# UNSTACK specific level
unstacked = stacked.unstack(1)
# Unstack only level 1 (keeps level 0 as index)

# UNSTACK at different levels
unstacked = stacked.unstack(0)
# Unstack level 0 instead

# FILL missing values after unstack
unstacked = stacked.unstack(fill_value=0)
# Fill NaN values with 0
```

**Mini-Example:**
```python
# Wide format (multiple columns)
wide = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'B'],
    'Q1': [100, 150, 200, 250],
    'Q2': [110, 160, 210, 260],
    'Q3': [120, 170, 220, 270]
})

# Stack: Convert quarters from columns to row index
# Q1, Q2, Q3 become index labels
stacked = wide.set_index('Category').stack()

# Unstack: Convert back to original (or different) format
long_format = stacked.unstack(0)  # Swap axes
# Now Category values become columns instead
```

**Study Notes:**
- `.stack()` - columns → row index (pivots down)
- `.unstack()` - row index → columns (pivots up)
- Creates/uses MultiIndex for multiple index levels
- Useful for converting between wide and long formats
- Common in time series and panel data

**Quick Tips:**
- Use after creating MultiIndex DataFrames
- `fill_value` parameter helpful for missing data after reshape
- Remember: stack makes data longer/narrower; unstack makes it wider
- Both return new DataFrames (don't modify original)

---

### 13.2 Pivot Tables

**Explanation:**
Create summary tables by specifying which columns become rows, columns, and values. Like Excel pivot tables.

**Code Examples:**

```python
# CREATE pivot table
pivot = pd.pivot_table(
    df,
    values='D',  # Which column to aggregate
    index=['A', 'B'],  # These become row index
    columns=['C'],  # These become columns
    aggfunc='mean'  # How to aggregate (default: mean)
)

# Different aggregation functions
pivot = pd.pivot_table(df, values='D', index='A', columns='B', aggfunc='sum')
pivot = pd.pivot_table(df, values='D', index='A', columns='B', aggfunc='count')

# Multiple aggregation functions
pivot = pd.pivot_table(
    df,
    values='D',
    index='A',
    columns='B',
    aggfunc=['mean', 'sum', 'count']
)

# Multiple value columns
pivot = pd.pivot_table(
    df,
    values=['D', 'E'],
    index='A',
    columns='B'
)

# Fill missing values
pivot = pd.pivot_table(df, values='D', index='A', columns='B', fill_value=0)
```

**Mini-Example:**
```python
# Sales data
sales = pd.DataFrame({
    'Month': ['Jan', 'Jan', 'Feb', 'Feb', 'Mar', 'Mar'],
    'Region': ['North', 'South', 'North', 'South', 'North', 'South'],
    'Amount': [100, 150, 120, 180, 110, 200]
})

# Create pivot table: Regions as columns, Months as rows
pivot = pd.pivot_table(
    sales,
    values='Amount',
    index='Month',
    columns='Region',
    aggfunc='sum'
)

# Result:
#         North  South
# Month
# Jan       100    150
# Feb       120    180
# Mar       110    200
```

**Study Notes:**
- `pivot_table()` creates summary tables
- `values` = what to aggregate
- `index` = row labels
- `columns` = column labels
- `aggfunc` = aggregation function (mean, sum, count, etc.)
- Similar to Excel Pivot Tables
- Automatically handles missing values with NaN

**Quick Tips:**
- Default aggfunc is 'mean'
- Use `fill_value` to replace NaN with a specific value
- Perfect for creating summary reports
- Can use multiple aggregation functions at once
- Great for exploratory data analysis

---

## SECTION 14: Time Series
---

### 14.1 Resampling

**Explanation:**
Convert time series data from one frequency to another. For example, convert daily data to monthly averages.

**Code Examples:**

```python
# CREATE time series
rng = pd.date_range('1/1/2024', periods=100, freq='D')  # 100 days
ts = pd.Series(range(100), index=rng)  # Values 0-99

# RESAMPLE to 5-day frequency
resampled = ts.resample('5D').sum()
# Sums values every 5 days
# '5D' = 5 days, 'M' = month, 'W' = week, 'H' = hour

# Different resampling functions
resampled = ts.resample('W').mean()  # Weekly average
resampled = ts.resample('M').sum()  # Monthly total
resampled = ts.resample('10D').max()  # Max value every 10 days

# UPSAMPLE to higher frequency (fill missing)
upsampled = ts.resample('12H').ffill()  # Forward fill every 12 hours
upsampled = ts.resample('12H').interpolate()  # Interpolate values

# Frequency strings:
# 'D' = day, 'W' = week, 'M' = month, 'Q' = quarter, 'Y' = year
# 'H' = hour, 'T' = minute, 'S' = second
# Prefix with number for multiples: '5D' = 5 days
```

**Mini-Example:**
```python
# Daily temperature readings for a month
dates = pd.date_range('2024-01-01', periods=30, freq='D')
temps = pd.Series([15+i*0.5 for i in range(30)], index=dates)

# Convert to weekly average
weekly_avg = temps.resample('W').mean()
# Output: Weekly average temperatures (approximately 4 weeks)

# Convert to every 5 days
every5 = temps.resample('5D').mean()
# Output: Average temperature every 5 days
```

**Study Notes:**
- `.resample()` changes time series frequency
- Downsampling (fewer points): use aggregation (sum, mean, etc.)
- Upsampling (more points): use filling/interpolation
- Common frequencies: D, W, M, Q, Y, H, T, S
- Can combine with `.agg()` for custom operations

**Quick Tips:**
- Common frequencies: 'D' (daily), 'M' (monthly), 'Q' (quarterly), 'Y' (yearly)
- Downsampling with `.sum()` → total; with `.mean()` → average
- Upsampling with `.ffill()` → repeat last value; with `.interpolate()` → estimate
- Perfect for financial and meteorological data

---

### 14.2 Time Zone Handling

**Explanation:**
Convert time-aware data between different time zones.

**Code Examples:**

```python
# CREATE time series without timezone
rng = pd.date_range('2024-01-01', periods=5, freq='D')
ts = pd.Series(range(5), index=rng)

# ADD timezone (localize)
ts_utc = ts.tz_localize('UTC')
# Now timezone-aware: 2024-01-01 00:00:00+00:00

# CONVERT to different timezone
ts_eastern = ts_utc.tz_convert('US/Eastern')
# Converts from UTC to Eastern time

# COMMON timezones:
# 'UTC' - Universal Coordinated Time
# 'US/Eastern', 'US/Central', 'US/Mountain', 'US/Pacific'
# 'Europe/London', 'Europe/Paris', 'Asia/Tokyo'
```

**Mini-Example:**
```python
# UTC times
utc_times = pd.date_range('2024-01-01', periods=3, freq='D')
ts = pd.Series([10, 20, 30], index=utc_times)

# Make timezone-aware (UTC)
ts_utc = ts.tz_localize('UTC')

# Convert to Tokyo time
ts_tokyo = ts_utc.tz_convert('Asia/Tokyo')
# Times shift forward (Tokyo is UTC+9)
```

**Study Notes:**
- `.tz_localize('timezone')` - add timezone to naive datetime
- `.tz_convert('timezone')` - convert between timezones
- Timezone-aware datetimes show offset (+00:00, -05:00, etc.)
- Use for international data handling

**Quick Tips:**
- Always work in UTC internally, convert for display
- Use `.tz_localize()` when you know the original timezone
- Use `.tz_convert()` after localizing to change timezone
- See pytz documentation for full list of timezones

---

## SECTION 15: Categorical Data
---

### 15.1 Working with Categories

**Explanation:**
Treat columns as categories for better memory usage and logical operations. Useful for columns with limited unique values (like grades, status, regions).

**Code Examples:**

```python
# CREATE DataFrame with grades
df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5, 6],
    'raw_grade': ['a', 'b', 'b', 'a', 'a', 'e']
})

# CONVERT to categorical
df['grade'] = df['raw_grade'].astype('category')
# Saves memory when many repeated values

# RENAME categories
new_categories = ['very good', 'good', 'very bad']
df['grade'] = df['grade'].cat.rename_categories(new_categories)

# SET category order
df['grade'] = df['grade'].cat.set_categories(
    ['very bad', 'bad', 'medium', 'good', 'very good']
)
# Now sorting uses this order, not alphabetical

# SORT by category order
sorted_df = df.sort_values(by='grade')
# Sorts by category order, not alphabetically

# GROUP BY including empty categories
grouped = df.groupby('grade', observed=False).size()
# Shows all categories, even those with zero occurrences
```

**Mini-Example:**
```python
# Survey responses
survey = pd.DataFrame({
    'Response': ['Very Satisfied', 'Neutral', 'Very Satisfied', 'Dissatisfied']
})

# Convert to ordered categories
survey['Response'] = survey['Response'].astype('category')
survey['Response'] = survey['Response'].cat.set_categories(
    ['Dissatisfied', 'Neutral', 'Very Satisfied'],
    ordered=True
)

print(survey['Response'].value_counts())
# Counts by category
```

**Study Notes:**
- Categorical = fixed set of possible values
- Memory-efficient for repeated values
- Use `.cat` accessor for category operations
- `.cat.rename_categories()` - change category names
- `.cat.set_categories()` - define order
- `observed=False` in groupby shows all categories

**Quick Tips:**
- Use categories for columns with ~100 or fewer unique values
- Define meaningful order for sorting (not alphabetical)
- Categories save memory and improve performance
- Useful for ordinal data (low, medium, high)

---

## SECTION 16: Importing and Exporting Data
---

### 16.1 Reading and Writing CSV

**Explanation:**
Load data from CSV files and save DataFrames to CSV.

**Code Examples:**

```python
# READ from CSV file
df = pd.read_csv('data.csv')
# Loads CSV file into DataFrame
# First row assumed to be column headers

# READ with specific settings
df = pd.read_csv('data.csv', sep=',', encoding='utf-8')  # Comma separator (default)
df = pd.read_csv('data.csv', sep=';')  # Semicolon separator
df = pd.read_csv('data.csv', index_col=0)  # First column as index

# SKIP rows
df = pd.read_csv('data.csv', skiprows=5)  # Skip first 5 rows
df = pd.read_csv('data.csv', nrows=100)  # Read only first 100 rows

# WRITE to CSV
df.to_csv('output.csv')
# Saves DataFrame to CSV file
# Index is included by default

# WRITE without index
df.to_csv('output.csv', index=False)
# Don't save row index

# WRITE specific columns
df.to_csv('output.csv', columns=['A', 'B', 'C'])
# Only save these columns
```

**Mini-Example:**
```python
# Create a DataFrame
sales = pd.DataFrame({
    'Product': ['A', 'B', 'C'],
    'Q1': [100, 150, 200],
    'Q2': [110, 160, 210]
})

# Save to CSV
sales.to_csv('sales.csv', index=False)
# File contents:
# Product,Q1,Q2
# A,100,110
# B,150,160
# C,200,210

# Read it back
df = pd.read_csv('sales.csv')
# Recreates the DataFrame
```

**Study Notes:**
- `read_csv()` - load CSV file
- `to_csv()` - save to CSV file
- CSV files are text-based (universal compatibility)
- Index is saved by default (use `index=False` to omit)
- Can specify separators, skip rows, etc.

**Quick Tips:**
- Always use `index=False` when exporting to avoid extra index column
- Use `sep=` parameter if CSV uses different delimiter (;, |, etc.)
- Use `nrows=` to read large files in chunks
- CSV is human-readable but not the most efficient format

---

### 16.2 Reading and Writing Excel

**Explanation:**
Work with Excel files (.xlsx, .xls format).

**Code Examples:**

```python
# READ from Excel file
df = pd.read_excel('data.xlsx')
# Reads first sheet by default

# READ specific sheet
df = pd.read_excel('data.xlsx', sheet_name='Sales')
# Read sheet named 'Sales'

df = pd.read_excel('data.xlsx', sheet_name=0)
# Read first sheet (index 0)

# WRITE to Excel
df.to_excel('output.xlsx', sheet_name='Sheet1')
# Creates Excel file with one sheet

# WRITE multiple sheets
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')
    df3.to_excel(writer, sheet_name='Sheet3')
# Creates file with multiple sheets
```

**Mini-Example:**
```python
# Read Excel file
df = pd.read_excel('sales_data.xlsx', sheet_name='2024_Sales')

# Process data
df_summary = df.groupby('Region').sum()

# Write to new Excel file
df_summary.to_excel('summary.xlsx', sheet_name='Summary')
```

**Study Notes:**
- `read_excel()` - load Excel file
- `to_excel()` - save to Excel file
- Can specify sheet by name or index
- `ExcelWriter` for multiple sheets
- Excel files preserve formatting (CSV doesn't)

**Quick Tips:**
- Use Excel for formatted reports
- Use CSV for data exchange/processing
- Multiple sheets make large files easier to manage
- Excel files are slower to read than CSV

---

### 16.3 Reading and Writing Parquet

**Explanation:**
Parquet is a modern, efficient binary format for data storage. Better for big data than CSV.

**Code Examples:**

```python
# WRITE to Parquet
df.to_parquet('data.parquet')
# Saves efficiently in binary format

# READ from Parquet
df = pd.read_parquet('data.parquet')
# Loads Parquet file back into DataFrame

# PARQUET advantages over CSV:
# - Smaller file size (compressed)
# - Faster read/write
# - Preserves data types
# - Efficient for large datasets
```

**Mini-Example:**
```python
# Create large DataFrame
large_df = pd.DataFrame({
    'ID': range(1000000),
    'Value': np.random.randn(1000000)
})

# Save as Parquet (smaller, faster)
large_df.to_parquet('large_data.parquet')

# Save as CSV (larger, slower)
large_df.to_csv('large_data.csv')

# Parquet file will be ~10x smaller and ~10x faster to load!
```

**Study Notes:**
- Parquet = binary format (not human-readable)
- More efficient than CSV
- Preserves all data types
- Better for big data and production systems
- Less compatible (fewer programs support it than CSV)

**Quick Tips:**
- Use Parquet for storing large datasets
- Use CSV for sharing/exchange
- Parquet is the industry standard for data lakes
- Requires `pyarrow` package (`pip install pyarrow`)

---

## SECTION 17: Common Pitfalls and Solutions
---

### 17.1 The "Gotcha" Guide

**Explanation:**
Avoid common mistakes when working with pandas.

**Common Issues:**

```python
# GOTCHA 1: Boolean Series in if statement
if pd.Series([False, True, False]):  # ERROR!
    print("Won't work")
# ValueError: The truth value of a Series is ambiguous

# SOLUTION:
if (pd.Series([False, True, False])).any():  # Check if ANY True
    print("This works")

if (pd.Series([False, True, False])).all():  # Check if ALL True
    print("This also works")

# GOTCHA 2: Chained indexing warning
df['col1']['row1'] = 10  # SettingWithCopyWarning!

# SOLUTION: Use .loc[] or .at[]
df.loc['row1', 'col1'] = 10  # Correct
df.at['row1', 'col1'] = 10  # Also correct

# GOTCHA 3: Using = instead of == in conditions
df[df['A'] = 5]  # SyntaxError!

# SOLUTION: Use == for comparison
df[df['A'] == 5]  # Correct

# GOTCHA 4: Need parentheses with multiple conditions
df[df['A'] > 0 & df['B'] < 100]  # Error!

# SOLUTION: Add parentheses
df[(df['A'] > 0) & (df['B'] < 100)]  # Correct

# GOTCHA 5: and/or instead of &/| with boolean Series
df[df['A'] > 0 and df['B'] < 100]  # Error!

# SOLUTION: Use & and |, not 'and' and 'or'
df[(df['A'] > 0) & (df['B'] < 100)]  # Correct

# GOTCHA 6: Modifying copy instead of original
df_copy = df[df['A'] > 0]
df_copy['B'] = 100  # May show SettingWithCopyWarning

# SOLUTION: Explicitly copy
df_copy = df[df['A'] > 0].copy()
df_copy['B'] = 100  # Now safe
```

**Mini-Example - Safe Modifications:**
```python
# Create DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [10, 20, 30]})

# WRONG: Chained indexing
# df['A'][0] = 999  # Warning!

# CORRECT: Use .at[] or .loc[]
df.at[0, 'A'] = 999  # Safe and fast
df.loc[0, 'A'] = 999  # Also safe

# WRONG: Boolean without condition check
# if df['A'] > 2:  # Error!

# CORRECT: Use .any() or .all()
if (df['A'] > 2).any():  # Check if any value > 2
    print("At least one value > 2")
```

**Study Notes:**
- Boolean Series can't be used directly in `if` (use `.any()` or `.all()`)
- Always use `.loc[]`, `.at[]`, `.iat[]` for safe modifications
- Use `==` not `=` for comparisons
- Use `&`, `|`, `~` for boolean operations (not `and`, `or`, `not`)
- Remember parentheses around each condition
- Use `.copy()` when creating subsets to modify

**Quick Tips:**
- Add `.copy()` when filtering to avoid warnings
- Use `.loc[]` or `.at[]` for all assignments (faster, safer)
- Test with small data first before big operations
- Check `.shape` and `.dtypes` after each major operation
- Use `.head()` to verify data after loading/transforming

---

## FINAL SUMMARY: KEY TAKEAWAYS
---

### Essential Methods Reference

**Data Creation:**
- `pd.Series()` - 1D array
- `pd.DataFrame()` - 2D table
- `pd.date_range()` - Date sequences
- `pd.concat()` - Combine DataFrames

**Data Inspection:**
- `.head()`, `.tail()` - Preview data
- `.shape`, `.dtypes`, `.info()` - Metadata
- `.describe()` - Statistics
- `.isnull()` - Missing values check

**Selection:**
- `df['column']` - Single column
- `df[['col1', 'col2']]` - Multiple columns
- `.loc[label]` - Selection by label
- `.iloc[position]` - Selection by position
- `df[condition]` - Boolean indexing

**Modification:**
- `df['column'] = values` - Add/update column
- `.at[]` - Set single value by label
- `.iat[]` - Set single value by position
- `.fillna()` - Handle missing values

**Analysis:**
- `.mean()`, `.sum()`, `.count()` - Aggregation
- `.groupby()` - Group operations
- `.value_counts()` - Frequency counts
- `.describe()` - Summary statistics

**Reshaping:**
- `.stack()` / `.unstack()` - Pivot data
- `.pivot_table()` - Create summaries
- `.resample()` - Time series frequency

**Input/Output:**
- `.read_csv()`, `.to_csv()` - CSV files
- `.read_excel()`, `.to_excel()` - Excel files
- `.read_parquet()`, `.to_parquet()` - Parquet files

### Quick Reference: When to Use What

| Task | Method |
|------|--------|
| Get one value | `.at[row, col]` |
| Get one row | `.loc[row_label]` or `.iloc[position]` |
| Get one column | `df['column']` |
| Filter rows | `df[condition]` (boolean indexing) |
| Sort data | `.sort_values(by=)` or `.sort_index()` |
| Group and summarize | `.groupby().agg()` |
| Handle missing data | `.fillna()` or `.dropna()` |
| Change data shape | `.stack()`, `.unstack()`, `.pivot_table()` |
| Combine DataFrames | `pd.concat()` or `pd.merge()` |

### Best Practices

1. **Always inspect first:** Use `.head()`, `.info()`, `.dtypes` immediately after loading
2. **Check for missing:** Use `.isnull().sum()` early and decide: drop or fill?
3. **Use .copy():** When filtering, add `.copy()` to avoid warnings
4. **Be explicit:** Use `.loc[]` and `.at[]` for modifications (safer than chaining)
5. **Verify results:** Use `.shape`, `.dtypes` after each transformation
6. **Use meaningful labels:** Named columns and index make code readable
7. **Document assumptions:** Comment about your data: what's missing, why
8. **Test with small data:** Verify logic before applying to full dataset

---

**Happy Pandas Learning!** 🐼📊
