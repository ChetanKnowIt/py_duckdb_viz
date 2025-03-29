
                    SELECT 
                        last_update,
                        COUNT(*) as count
                    FROM {{ table_name }}
                    GROUP BY 1
                    ORDER BY 1
                    LIMIT {{ limit }}
                