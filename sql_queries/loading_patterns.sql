
                    SELECT 
                        top_level_hostname,
                        resource_hostname,
                        resource_type,
                        COUNT(*) as frequency
                    FROM {{ table_name }}
                    GROUP BY 1, 2, 3
                    ORDER BY 4 DESC
                    LIMIT {{ limit }}
                