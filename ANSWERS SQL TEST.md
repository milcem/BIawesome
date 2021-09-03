=======================================================================
1. Suppose you’ve created a table called ‘menu’ using below SQL query:
=======================================================================

CREATE TABLE menu ( dish_id SERIAL PRIMARY KEY, name varchar);

Answer:
--------

menu

dish_id  |   name
-------------------------
	1	 |	  Fish
	2	 |	  Beef
	3    |    Veggie
	4	 |    Mushrooms


Which of the following will be the output of the below query?

Select * FROM menu;


Answer:
--------


dish_id  |   name
-------------------------
	1	 |	  Fish
	2	 |	  Beef
	3    |    Veggie
	4	 |    Mushrooms


===================================================================================
2. Suppose you are given a table ‘words’. The table has 2 columns ‘id’ and ‘word’.
===================================================================================


‘id’	|	word
-----------------
 1		|	My
 2		|   Name
 3		|   Is
 4		|   Ankit
 5		|   Gupta
 6      |   I
 7		|   Love
 8 		|   Solving
 9      |   Data
 10		|   Mining
 11     |   Problems


What will be the output for the below query?
select c1, c2, c3 from 
( select id,  
lag(word) over (order by id) as c1, 
word as c2, 
lead(word) over (order by id) as c3 from words ) as t where c2 = ‘Mining’ or c2 = ‘Problems’;


lag(word) over (order by id) as c1, 

‘id’	|	‘word’	|	c1
-------------------------------
 1		|	My		|	/
 2		|   Name	|	My
 3		|   Is 		|	Name
 4		|   Ankit	|   Is
 5		|   Gupta	|	Ankit
 6      |   I 		|   Gupta
 7		|   Love    |   I
 8 		|   Solving |   Love
 9      |   Data 	|	Solving
 10		|   Mining  |   Data
 11     |   Problems|   Mining

word as c2, 

‘id’	|	‘word’	|	c1 		 |	c2
-----------------------------------------
 1		|	My		|	/		 |  My
 2		|   Name	|	My 		 |  Name
 3		|   Is 		|	Name 	 |  Is
 4		|   Ankit	|   Is 		 |  Ankit		
 5		|   Gupta	|	Ankit 	 |  Gupta
 6      |   I 		|   Gupta	 |  I
 7		|   Love    |   I 		 |  Love
 8 		|   Solving |   Love 	 |  Solving
 9      |   Data 	|	Solving  |  Data
 10		|   Mining  |   Data 	 |  Mining
 11     |   Problems|   Mining   |  Problems


 lead(word) over (order by id) as c3 from words ) 



‘id’	|	‘word’	|	c1 		 |	c2		|  c3
-----------------------------------------------------
 1		|	My		|	/		 |  My		| Name
 2		|   Name	|	My 		 |  Name    | Is
 3		|   Is 		|	Name 	 |  Is      | Ankit
 4		|   Ankit	|   Is 		 |  Ankit	| Gupta
 5		|   Gupta	|	Ankit 	 |  Gupta   | I
 6      |   I 		|   Gupta	 |  I       | Love
 7		|   Love    |   I 		 |  Love    | Solving
 8 		|   Solving |   Love 	 |  Solving | Data
 9      |   Data 	|	Solving  |  Data    | Mining
 10		|   Mining  |   Data 	 |  Mining  | Problems 
 11     |   Problems|   Mining   |  Problems| \


select c1, c2, c3 from ( select id,  lag(word) over (order by id) as c1, word as c2, lead(word) over (order by id) as c3 from words ) as t where c2 = ‘Mining’ or c2 = ‘Problems’;

Answer:
--------

   c1    |    c2     |    c3
 --------|-----------|-------------     
Data 	 |  Mining   | Problems 
Mining   |  Problems | \

================================================================================================================================================================
3. Suppose you have a CSV file which has 3 columns (‘User_ID’, ‘Gender’, ‘product_ID’) and 7150884 rows. You have created a table “train” from this file in SQL.
Now, you run Query 1 (mentioned below):
EXPLAIN SELECT * from train WHERE product_ID like '%7085%';
Then, you created product_ID columns as an index in ‘train’ table using below SQL query:
CREATE INDEX product_ID ON train(Product_ID)
Suppose, you run Query 2 (same as Query 1) on train table.
EXPLAIN SELECT * from train WHERE product_ID like '%7085%';
Let T1 and T2 be time taken by Query 1 and Query 2 respectively. Which query will take less time to execute?
=================================================================================================================================================================

Answer:
--------
Creating an index will speed up the SQL query as it would create a column indexing just 'product_ID' within the 'train' table as then it would only search through the indexed column product_ID rather than looking at every entry in the 'train' table. So Query 2 would take less time to execute. 


============================================================================================================================================================================
4. Indexing is useful in a database for fast searching. Generally, B-tree is used for indexing in a database. 
Now, you want to use Binary Search Tree instead of B-tree.
Suppose there are numbers between 1 and 100 and you want to search the number 35 using Binary Search Tree algorithm. Which of the following sequences CANNOT be the sequence for the numbers examined?
============================================================================================================================================================================

A. 10, 75, 64, 43, 60, 57, 55
B. 90, 12, 68, 34, 62, 45, 55
C. 9, 85, 47, 68, 43, 57, 55
D. 79, 14, 72, 56, 16, 53, 55


A.						B.						C. 							D.
						
   10 						90 						9 								79
   /\						/\ 						/\								/\
     75                   12						 85 						  14
	/\ 					   /\						  /\ 						  /\
  64                        68                      47 								72
   /\                        /\                     /\ 								/\
 43                        34 						  68 					      56
  /\                       /\						  /\						  /\
    60                       62                  (x)43                          16
   /\						/\					  /\ 							/\
 57 					  45                        57 							  53 
 /\                      /\							 /\                            /\
55                         55                      55                                55

POSSIBLE SEQ.		  POSSIBLE SEQ.				IMPOSSIBLE SEQ.                POSSIBLE SEQ.


Answer:
--------

C. Cannot be the sequence of numbers examined because at 3rd level, 68 branches out on the right hand side of 47 above, so every number spanned on the BST below HAS to be bigger than 47, yet we encounter 43 on the left hand side below 68.


=========================================================================================================================================
5. Consider the following relational schema.
Students(rollno: integer, sname: string)
Courses (courseno: integer, cname: string)
Registration (rollno: integer, courseno: integer, percent: real)
Now, which of the following query would be able to find the unique names of all students having score more than 90% in the courseno 107?
=========================================================================================================================================

A. SELECT DISTINCT S.sname FROM Students as S, Registration as R WHERE R.rollno=S.rollno AND R.courseno=107 AND R.percent >90
B. SELECT UNIQUE S.sname FROM Students as S, Registration as R WHERE R.rollno=S.rollno AND R.courseno=107 AND R.percent >90
C. SELECT sname FROM Students as S, Registration as R WHERE R.rollno=S.rollno AND R.courseno=107 AND R.percent >90
D. None of these


Answer:
--------

A. This SQL Query will first perform a cross product from Students (S) and Registration (R), WHERE only the rows in that cross product of students registered for course 107 are queried AND only those with a score more than 90%. DISTINCT helps us query only the distinct values (omitting duplicates).

If we used Oracle SQL, B. would provide the same result, but A is syntactically more correct as it's standard SQL, and UNIQUE is currently used as a database constraint when creating or updating the database, not when fetching results - for that we would use DISTINCT.


