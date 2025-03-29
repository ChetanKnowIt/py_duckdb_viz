:: we use DuckDB CLI with .mode html and .output results.html 
:: .mode sets the type of format we want to get
:: .output returns the formatted query result in terminal's standard output 
:: @echo off
:: (
	:: echo .mode html
    :: echo .output results.html
    :: echo SELECT * FROM redirect_statistics;
:: ) |  "E:\projects\DataAnalytics\SQL\duckdb\duckdb_cli-windows-amd64-DuckDB 1 0 0 Nivis\duckdb.exe" E:\projects\DataAnalytics\SQL\duckdb\load_statistics.db


:: lets try other output formats 


:: if you give ./output/file.jsonl it won't create new ./output directory, CLI expects folder to exist 

::@echo off
::(
::	echo .mode jsonlines
::    echo .output ./output/results.jsonl
::    echo SELECT * FROM redirect_statistics;
::) |  "E:\projects\DataAnalytics\SQL\duckdb\duckdb_cli-windows-amd64-DuckDB 1 0 0 Nivis\duckdb.exe" E:\projects\DataAnalytics\SQL\duckdb\load_statistics.db
::

:: REM OUTPUT: writing markdown..
::@echo off
::(
::	echo .mode markdown
::    echo .output ./output/results.md
::    echo SELECT * FROM redirect_statistics;
::) |  "E:\projects\DataAnalytics\SQL\duckdb\duckdb_cli-windows-amd64-DuckDB 1 0 0 Nivis\duckdb.exe" E:\projects\DataAnalytics\SQL\duckdb\load_statistics.db

:: REM OUTPUT: writing csv..
::@echo off
::(
::	echo .mode csv
::    echo .output ./output/results.csv
::    echo SELECT * FROM redirect_statistics;
::) |  "E:\projects\DataAnalytics\SQL\duckdb\duckdb_cli-windows-amd64-DuckDB 1 0 0 Nivis\duckdb.exe" E:\projects\DataAnalytics\SQL\duckdb\load_statistics.db

REM OUTPUT: writing INSERT SQL....
@echo off
(
	echo .mode insert
    echo .output ./output/results.txt
    echo SELECT * FROM redirect_statistics;
) |  "E:\projects\DataAnalytics\SQL\duckdb\duckdb_cli-windows-amd64-DuckDB 1 0 0 Nivis\duckdb.exe" E:\projects\DataAnalytics\SQL\duckdb\load_statistics.db
