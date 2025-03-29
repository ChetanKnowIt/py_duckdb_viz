import matplotlib
# Set the backend to 'Agg' before importing pyplot
# This prevents Tkinter-related errors
matplotlib.use('Agg')

import argparse
import logging
import os
import sys
from pathlib import Path
import json
from functools import lru_cache
import warnings
import concurrent.futures

# Global configuration
INPUT_CONFIG = {
    'db': {
        'default': 'database.db',
        'desc': 'Path to the database file'
    },
    'table': {
        'default': 'load_statistics',
        'desc': 'Name of the table to analyze'
    },
    'templates': {
        'default': 'templates.yml',
        'desc': 'Path to query configuration file (YAML or JSON)'
    },
    'html_template': {
        'default': 'report_template.html',
        'desc': 'Path to HTML report template'
    },
    'output': {
        'default': 'output_images',
        'desc': 'Directory for saving visualizations and reports'
    },
    'report': {
        'default': 'analysis_report.html',
        'desc': 'Name of the HTML report file'
    }
}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze SQLite/DuckDB database with customizable SQL queries')
    
    # Original arguments
    parser.add_argument('--db_path', required=True, help=INPUT_CONFIG['db']['desc'])
    parser.add_argument('--table_name', default=INPUT_CONFIG['table']['default'], help=INPUT_CONFIG['table']['desc'])
    parser.add_argument('--query_config', default=INPUT_CONFIG['templates']['default'], help=INPUT_CONFIG['templates']['desc'])
    parser.add_argument('--html_template', default=INPUT_CONFIG['html_template']['default'], help=INPUT_CONFIG['html_template']['desc'])
    parser.add_argument('--output_dir', default=INPUT_CONFIG['output']['default'], help=INPUT_CONFIG['output']['desc'])
    parser.add_argument('--report_name', default=INPUT_CONFIG['report']['default'], help=INPUT_CONFIG['report']['desc'])
    parser.add_argument('--embed_images', action='store_true', help='Embed images in HTML report instead of linking')
    
    # Performance-related arguments
    parser.add_argument('--threads', type=int, default=1, help='Number of database threads to use')
    parser.add_argument('--parallel', action='store_true', help='Process queries in parallel')
    parser.add_argument('--cache_queries', action='store_true', help='Cache query results')
    parser.add_argument('--skip_viz', action='store_true', help='Skip generating visualizations')
    parser.add_argument('--validate_only', action='store_true', help='Validate configuration without executing')
    
    parser.add_argument('--log_level', default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    
    return parser.parse_args()

def setup_logger(log_level=logging.INFO):
    """Configure logging"""
    logger = logging.getLogger("db_analyzer")
    logger.setLevel(log_level)
    
    # Clear existing handlers to prevent duplicate logging
    if logger.handlers:
        logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    
    return logger

class QueryManager:
    """Class to manage SQL queries with templating and parameter validation"""
    
    def __init__(self, config_path, use_cache=False):
        """Initialize QueryManager with config file path"""
        self.queries = {}
        self.query_cache = {} if use_cache else None
        self.logger = logging.getLogger("db_analyzer.query_manager")
        self.load_from_file(config_path)
    
    def load_from_file(self, path):
        """Load queries from YAML/JSON configuration file"""
        self.logger.info(f"Loading queries from {path}")
        try:
            with open(path, 'r') as f:
                if path.endswith('.yaml') or path.endswith('.yml'):
                    import yaml
                    self.queries = yaml.safe_load(f)
                else:
                    self.queries = json.load(f)
            
            self.logger.info(f"Loaded {len(self.queries)} queries from configuration")
        except Exception as e:
            self.logger.error(f"Error loading query configuration: {e}")
            raise ValueError(f"Failed to load query configuration: {e}")
    
    def _validate_parameters(self, query_def, params):
        """Validate parameters against query definition"""
        if 'parameters' not in query_def:
            return True
        
        for param_name, param_spec in query_def['parameters'].items():
            if param_spec.get('required', False) and param_name not in params:
                self.logger.error(f"Missing required parameter: {param_name}")
                return False
        
        return True
    
    def _process_params(self, query_def, params):
        """Process parameters by applying defaults"""
        processed = params.copy() if params else {}
        
        if 'parameters' not in query_def:
            return processed
            
        # Apply defaults for missing parameters
        for param_name, param_spec in query_def['parameters'].items():
            if param_name not in processed and 'default' in param_spec:
                processed[param_name] = param_spec['default']
        
        return processed
    
    def get_query_names(self):
        """Return list of available query names"""
        return list(self.queries.keys())
    
    def get_query_info(self, query_name):
        """Return metadata about a specific query"""
        if query_name not in self.queries:
            return None
        
        query_def = self.queries[query_name]
        return {
            'title': query_def.get('title', query_name),
            'has_visualization': 'visualization' in query_def
        }
    
    def execute(self, conn, query_name, params=None):
        """Execute a named query with parameters"""
        if params is None:
            params = {}
            
        if query_name not in self.queries:
            self.logger.error(f"Unknown query: {query_name}")
            return None

        # Check cache if enabled
        if self.query_cache is not None:
            cache_key = f"{query_name}_{hash(frozenset(params.items() if params else {}))}"
            if cache_key in self.query_cache:
                self.logger.debug(f"Using cached result for query '{query_name}'")
                return self.query_cache[cache_key]
            
        query_def = self.queries[query_name]
        
        # Validate parameters
        if not self._validate_parameters(query_def, params):
            return None
        
        # Process parameters (apply defaults)
        processed_params = self._process_params(query_def, params)
        
        # Render the query template
        try:
            import jinja2
            template = jinja2.Template(query_def['sql'])
            sql = template.render(**processed_params)
            
            self.logger.debug(f"Executing query '{query_name}': {sql}")
            
            # Execute with appropriate error handling
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = conn.execute(sql).fetchdf()
                
            self.logger.info(f"Query '{query_name}' returned {len(result)} rows")
            
            # Cache the result if caching is enabled
            if self.query_cache is not None:
                self.query_cache[cache_key] = result
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing query '{query_name}': {e}")
            return None

def create_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    logger = logging.getLogger("db_analyzer")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory '{output_dir}'")
    else:
        logger.info(f"Using existing directory '{output_dir}'")

def visualize_bar_chart(data, x_col, y_col, title, filename, output_dir, palette=None, horizontal=False):
    """Create and save a bar chart visualization"""
    logger = logging.getLogger("db_analyzer")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if data is None or data.empty:
            logger.warning(f"No data available for visualization: {title}")
            return False
        
        plt.figure(figsize=(12, 8))
        
        # Apply a professional theme
        sns.set_theme(style="whitegrid")
        
        if horizontal:
            # Horizontal bar chart - using the older API to match original
            if palette:
                ax = sns.barplot(x=x_col, y=y_col, data=data, palette=palette)
            else:
                ax = sns.barplot(x=x_col, y=y_col, data=data)
            plt.tight_layout()
        else:
            # Vertical bar chart - using the older API to match original
            if palette:
                ax = sns.barplot(x=x_col, y=y_col, data=data, palette=palette)
            else:
                ax = sns.barplot(x=x_col, y=y_col, data=data)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
        
        plt.title(title)
        
        # Save the figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved as '{output_path}'")
        return output_path
    
    except Exception as e:
        logger.error(f"Visualization error for {title}: {e}")
        return False

def generate_html_report(results, output_dir, html_template_path, report_name='analysis_report.html', embed_images=False):
    """Generate an HTML report with all query results and visualizations"""
    logger = logging.getLogger("db_analyzer")
    try:
        from datetime import datetime
        import jinja2
        import base64
        
        report_path = os.path.join(output_dir, report_name)
        logger.info(f"Generating HTML report: {report_path}")
        
        # Load HTML template
        with open(html_template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # Create Jinja2 template
        template = jinja2.Template(template_content)
        
        # Process results for template rendering
        template_data = {
            'db_name': os.path.basename(results.get('db_path', 'Unknown')),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'completion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'results': {},
            'embed_images': embed_images
        }
        
        # Process each result
        for name, data in results.items():
            if name == 'db_path':
                continue
                
            template_data['results'][name] = {
                'title': data.get('title', name)
            }
            
            # Process DataFrame
            if data.get('dataframe') is not None:
                df = data['dataframe']
                if not df.empty:
                    template_data['results'][name]['dataframe'] = df
                    template_data['results'][name]['dataframe_html'] = df.to_html(index=False, classes="dataframe", border=0)
            
            # Process visualization
            if data.get('visualization'):
                template_data['results'][name]['visualization'] = True
                
                if embed_images and os.path.exists(data["visualization"]):
                    # Embed the image directly in the HTML using base64
                    try:
                        img_format = os.path.splitext(data["visualization"])[1][1:].lower()
                        if img_format == 'jpg':
                            img_format = 'jpeg'
                        
                        with open(data["visualization"], "rb") as img_file:
                            img_data = base64.b64encode(img_file.read()).decode('utf-8')
                            
                        template_data['results'][name]['img_format'] = img_format
                        template_data['results'][name]['img_data'] = img_data
                    except Exception as e:
                        logger.error(f"Error embedding image {data['visualization']}: {e}")
                        viz_path = os.path.relpath(data["visualization"], os.path.dirname(report_path))
                        template_data['results'][name]['viz_path'] = viz_path
                else:
                    viz_path = os.path.relpath(data["visualization"], os.path.dirname(report_path))
                    template_data['results'][name]['viz_path'] = viz_path
            
            # Add error message if available
            if data.get('error'):
                template_data['results'][name]['error'] = data['error']
        
        # Render the template
        html_content = template.render(**template_data)
        
        # Write the HTML report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"HTML report generated successfully: {report_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating HTML report: {e}")
        return False

def process_query(conn, query_manager, query_name, base_params, output_dir, skip_viz=False):
    """Process a single query - used for both serial and parallel execution"""
    logger = logging.getLogger("db_analyzer")
    
    query_info = query_manager.get_query_info(query_name)
    if not query_info:
        return query_name, {"error": "Invalid query configuration"}
        
    title = query_info['title']
    logger.info(f"Processing query: {title}")
    
    # Initialize result 
    result = {
        "title": title
    }
    
    # Execute query
    query_result = query_manager.execute(conn, query_name, base_params)
    
    if query_result is not None:
        # Store dataframe result
        result["dataframe"] = query_result
        
        # Special processing for timeline data
        if not skip_viz and query_name == 'timeline' and 'last_update' in query_result.columns:
            try:
                # Import needed modules
                import duckdb
                import pandas as pd
                
                # Convert timestamps
                logger.info("Converting timeline timestamps...")
                
                # Create a temporary connection for the conversion
                temp_conn = duckdb.connect(':memory:')
                
                # Register the DataFrame as a virtual table
                temp_conn.register('df', query_result)
                
                # Execute the conversion query
                conversion_result = temp_conn.execute("""
                    SELECT 
                        CASE 
                            WHEN last_update IS NOT NULL THEN
                                STRFTIME(
                                    TIMESTAMP '1601-01-01 00:00:00' + 
                                    INTERVAL (last_update / 86400000000) DAY +
                                    INTERVAL ((last_update % 86400000000) / 1000) MILLISECOND,
                                    '%H_%M_%d%m%Y'
                                )
                            ELSE NULL
                        END AS converted_timestamp
                    FROM df
                """).fetchdf()
                
                # Close the temporary connection
                temp_conn.close()
                
                # Add the converted timestamps to the original DataFrame
                query_result['converted_timestamp'] = conversion_result['converted_timestamp'].values
                
                # Create a visualization for the timeline
                try:
                    # Extract just date part for grouping (remove time part)
                    query_result['date_only'] = query_result['converted_timestamp'].str.split('_').str[-1]
                    
                    # Group by date and sum counts
                    timeline_data = query_result.groupby('date_only').sum().reset_index()
                    timeline_data = timeline_data.sort_values('date_only')
                    
                    # Generate visualization
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    
                    plt.figure(figsize=(14, 8))
                    sns.set_theme(style="whitegrid")
                    
                    # Create line plot
                    ax = sns.lineplot(x='date_only', y='count', data=timeline_data, marker='o', linewidth=2.5)
                    
                    # Customize the plot
                    plt.title('Resource Updates Over Time', fontsize=16)
                    plt.xlabel('Date (DDMMYYYY)', fontsize=12)
                    plt.ylabel('Number of Resources', fontsize=12)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    # Save the figure
                    viz_filename = 'timeline_chart.png'
                    viz_path = os.path.join(output_dir, viz_filename)
                    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Add visualization to the results but remove the dataframe
                    result.pop("dataframe", None)
                    result["visualization"] = viz_path
                    logger.info(f"Timeline visualization saved as '{viz_path}'")
                    
                except Exception as e:
                    logger.error(f"Error creating timeline visualization: {e}")
                    
            except Exception as e:
                logger.error(f"Error converting timeline timestamps: {e}")
        
        # Generate visualization if specified in query config
        if not skip_viz and query_info.get('has_visualization', False):
            viz_config = query_manager.queries[query_name].get('visualization', {})
            if viz_config:
                viz_filename = f'{query_name}.png'
                
                # Get visualization parameters
                x_col = viz_config.get('x_col')
                y_col = viz_config.get('y_col')
                horizontal = viz_config.get('horizontal', False)
                palette = viz_config.get('palette')
                
                if x_col and y_col:
                    viz_path = visualize_bar_chart(
                        query_result, x_col, y_col,
                        title, viz_filename, output_dir,
                        palette=palette, horizontal=horizontal
                    )
                    if viz_path:
                        result["visualization"] = viz_path
    else:
        # Query failed
        result["error"] = f"Failed to execute query: {query_name}"
    
    return query_name, result

# Modified analyze_database function to fix visualization issues in parallel mode
def analyze_database(db_path, query_manager, output_dir, html_template_path, 
                    table_name="load_statistics", report_name="analysis_report.html", 
                    embed_images=False, skip_viz=False, use_parallel=False, threads=1):
    """Main function to analyze the database"""
    logger = logging.getLogger("db_analyzer")
    
    try:
        # Connect to the database
        import duckdb
        logger.info(f"Connecting to database: {db_path}")
        conn = duckdb.connect(db_path)
        
        # Set connection parameters
        conn.execute(f"SET threads TO {threads}")  # Use specified thread count
        logger.info(f"Database connection established with {threads} threads")
        
        # Base parameters for the SQL templates
        base_params = {
            'table_name': table_name  # Use provided table name
        }
        
        # Dictionary to store all results for the HTML report
        all_results = {
            'db_path': db_path
        }
        
        # Get all available queries
        query_names = query_manager.get_query_names()
        
        # Make sure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        # Decide on parallel or serial processing
        if use_parallel and threads > 1:
            logger.info(f"Processing queries in parallel with {min(len(query_names), threads)} workers")
            
            # Create a function for parallel processing that uses its own connection
            def parallel_process_query(query_name):
                # Create a new connection for thread safety
                thread_conn = duckdb.connect(db_path)
                thread_conn.execute(f"SET threads TO 1")  # Single-threaded for each worker
                
                result = process_query(thread_conn, query_manager, query_name, base_params, output_dir, skip_viz)
                
                # Close connection to release resources
                thread_conn.close()
                return result
            
            # Process queries in parallel - ensuring completion before report generation
            results_dict = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                future_to_query = {executor.submit(parallel_process_query, name): name for name in query_names}
                
                for future in concurrent.futures.as_completed(future_to_query):
                    query_name, result = future.result()
                    results_dict[query_name] = result
            
            # Important: Add results to all_results after all threads have completed
            # This ensures all visualizations are ready before report generation
            for query_name, result in results_dict.items():
                all_results[query_name] = result
                
            # Add a slight delay to ensure all file writes are complete
            import time
            time.sleep(1)
        else:
            logger.info("Processing queries serially")
            # Process each query
            for query_name in query_names:
                query_name, result = process_query(conn, query_manager, query_name, base_params, output_dir, skip_viz)
                all_results[query_name] = result
        
        # Close the connection
        conn.close()
        logger.info("Database connection closed")
        
        # Generate HTML report with all results
        generate_html_report(all_results, output_dir, html_template_path, report_name, embed_images)
        
        return True
    
    except Exception as e:
        logger.error(f"Database analysis error: {e}")
        return False

def validate_configuration(db_path, query_config_path):
    """Quickly validate database and query configurations"""
    logger = logging.getLogger("db_analyzer")
    
    try:
        import duckdb
        
        # Test database connection
        logger.info(f"Testing connection to database: {db_path}")
        conn = duckdb.connect(db_path)
        
        # Set connection parameters for better performance
        conn.execute("PRAGMA memory_limit='2GB'")  # Use more memory for better performance
        conn.execute("PRAGMA threads=4")  # Use multiple threads for each query
        # Test query configuration
        logger.info(f"Testing query configuration: {query_config_path}")
        query_manager = QueryManager(query_config_path)
        logger.info(f"Query configuration validated: {len(query_manager.get_query_names())} queries found")
        
        # Close connection
        conn.close()
        
        return True
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False

def main():
    """Main entry point"""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Setup logging
        global logger
        log_level = getattr(logging, args.log_level)
        logger = setup_logger(log_level)
        
        logger.info("Starting database analysis")
        
        # Just validate if requested
        if args.validate_only:
            if validate_configuration(args.db_path, args.query_config):
                logger.info("Configuration validation successful")
                return 0
            else:
                logger.error("Configuration validation failed")
                return 1
        
        # Create output directory
        create_output_directory(args.output_dir)
        
        # Initialize query manager with config file
        query_manager = QueryManager(args.query_config, use_cache=args.cache_queries)
        
        # Analyze the database
        success = analyze_database(
            args.db_path, 
            query_manager, 
            args.output_dir,
            args.html_template,
            args.table_name, 
            args.report_name, 
            args.embed_images,
            args.skip_viz,
            args.parallel,
            args.threads
        )
        
        if success:
            logger.info("Analysis completed successfully")
            logger.info(f"HTML report available at: {os.path.join(args.output_dir, args.report_name)}")
        else:
            logger.warning("Analysis completed with errors")
        
        return 0
    
    except Exception as e:
        if 'logger' in globals():
            logger.critical(f"Unhandled exception: {e}", exc_info=True)
        else:
            print(f"Critical error before logger initialization: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())