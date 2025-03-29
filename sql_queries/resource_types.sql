
                    SELECT 
                        resource_type,
                        COUNT(*) as count 
                    FROM {{ table_name }}
                    GROUP BY 1
                    ORDER BY 1
                