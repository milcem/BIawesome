Having this table **t1**

    id    |   ID Name   |   Parent ID
    1     |     A       |       4
    2     |     B       |       0
    3     |     C       |       2
    4     |     D       |       6
    5     |     E       |       3
    6     |     F       |       0
  
  

Create a view that enriche the table with the heightst hierarchy parent.


1. SQL code to create example table:

  -- table per excel example
  CREATE TABLE t1 (id int identity (1,1), IDname varchar(1), ParentID int)
  INSERT INTO t1 (IDname,ParentID) VALUES ('A',4)
  INSERT INTO t1 (IDname,ParentID) VALUES ('B',0)
  INSERT INTO t1 (IDname,ParentID) VALUES ('C',2)
  INSERT INTO t1 (IDname,ParentID) VALUES ('D',6)
  INSERT INTO t1 (IDname,ParentID) VALUES ('E',3)
  INSERT INTO t1 (IDname,ParentID) VALUES ('F',0)


  SELECT * FROM t1; 
  
<img width="468" alt="Picture1" src="https://user-images.githubusercontent.com/26208356/132016190-aee5318b-5a6c-43c2-8fc6-a86b38b86590.png">


2. Enriche table with hierarchy


  -- enrich t1 with heightst hierarchy parent function
      CREATE FUNCTION dbo.hierarchy(@id int)
      RETURNS VARCHAR(1)

      BEGIN

        DECLARE @pid AS int, @name varchar(1), @idnew int -- Declaring our temp variables
        SET @pid=100
        SET @idnew=@id

        WHILE @pid<>0
        BEGIN
          SELECT @pid=parentid, @name=IDname, @idnew=parentid
          FROM t1
          WHERE id=@idnew	
        END

        RETURN @name
      END
  
3. Create a view

        -- create a view showing enriched t1
        CREATE VIEW dbo.testHierarchy AS
        SELECT id, idname AS [ID Name], parentid AS [Parent Id], dbo.hierarchy(id) AS [id2] 
        FROM t1
        GO;


-

      SELECT * FROM t1;
      
      
<img width="468" alt="Picture2" src="https://user-images.githubusercontent.com/26208356/132016246-d530b254-c919-4bac-a1e2-5a0d2b5d2719.png">


      SELECT * FROM testHierarchy;
  
<img width="468" alt="Picture3" src="https://user-images.githubusercontent.com/26208356/132016275-87156fe5-8b7a-4e05-9f48-677ad711f4c5.png">

