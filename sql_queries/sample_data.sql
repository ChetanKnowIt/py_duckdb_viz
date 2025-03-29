
                    SELECT 
                        top_level_hostname,
                        resource_hostname,
                        resource_type,
                        last_update
                    FROM {{ table_name }}
                    LIMIT {{ limit }}
                