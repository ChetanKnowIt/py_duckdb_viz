
                    SELECT 
                        resource_hostname,
                        COUNT(*) as count 
                    FROM {{ table_name }}
                    GROUP BY 1
                    ORDER BY 2 DESC
                    LIMIT {{ limit }}
                