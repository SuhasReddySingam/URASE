import os
import sqlite3
import sqlparse
from sqlparse.sql import Statement, Token, TokenList
from sqlparse.tokens import Keyword, Name, Literal, Operator, Punctuation
import ast
from typing import List, Dict, Tuple, Set, Any, Optional
import numpy as np
from dataclasses import dataclass
import re
from function import validate_sql_statement
from arg import main_args

args = main_args()

@dataclass
class SQLCandidate:
    """Data class for SQL candidate queries"""
    sql: str
    index: int
    
    
@dataclass
class EvaluationScore:
    """Data class for evaluation scores"""
    executability: float  # Executability score (0 or 1)
    schema_conformity: float  # Schema conformity score (0-1)
    example_consistency: float  # Example consistency score (0-1)
    semantic_execution_consistency: float  # Semantic execution consistency score (0-1)
    

class ASTProcessor:
    """AST processor for calculating abstract syntax tree edit distance of SQL queries"""
    
    @staticmethod
    def parse_sql_to_ast(sql: str) -> Dict:
        """Parse SQL into simplified AST representation"""
        if not sql or not sql.strip():
            return {'type': 'Empty', 'value': '', 'tokens': []}
            
        try:
            parsed_statements = sqlparse.parse(sql)
            if not parsed_statements:
                return {'type': 'Empty', 'value': '', 'tokens': []}
                
            # Take the first statement
            statement = parsed_statements[0]
            return ASTProcessor._build_ast_dict(statement)
        except Exception as e:
            # Return empty AST when parsing fails
            return {'type': 'Error', 'value': str(e), 'tokens': []}
    
    @staticmethod
    def _build_ast_dict(token) -> Dict:
        """Recursively build AST dictionary"""
        if token is None:
            return {'type': 'None', 'value': '', 'tokens': []}
            
        # Get basic token information
        token_type = type(token).__name__
        token_value = str(token).strip()
        
        # If it's a TokenList type (contains sub-tokens), process recursively
        if hasattr(token, 'tokens') and token.tokens:
            child_tokens = []
            for sub_token in token.tokens:
                # Skip whitespace and meaningless tokens
                if ASTProcessor._is_meaningful_token(sub_token):
                    child_ast = ASTProcessor._build_ast_dict(sub_token)
                    if child_ast['type'] != 'None':  # Only add meaningful child nodes
                        child_tokens.append(child_ast)
            
            return {
                'type': token_type,
                'value': token_value,
                'ttype': str(token.ttype) if hasattr(token, 'ttype') and token.ttype else None,
                'tokens': child_tokens
            }
        else:
            # Leaf node
            return {
                'type': token_type,
                'value': token_value,
                'ttype': str(token.ttype) if hasattr(token, 'ttype') and token.ttype else None,
                'tokens': []
            }
    
    @staticmethod
    def _is_meaningful_token(token) -> bool:
        """Determine if token is meaningful (filter out whitespace and other useless tokens)"""
        if token is None:
            return False
            
        token_str = str(token).strip()
        if not token_str:
            return False
            
        # Filter out pure whitespace and punctuation (except some important ones)
        if hasattr(token, 'ttype'):
            if token.ttype in (sqlparse.tokens.Whitespace, 
                              sqlparse.tokens.Whitespace.Newline,
                              sqlparse.tokens.Comment.Single,
                              sqlparse.tokens.Comment.Multiline):
                return False
        
        return True
    
    @staticmethod
    def calculate_edit_distance(ast1: Dict, ast2: Dict) -> float:
        """Calculate normalized edit distance between two ASTs"""
        
        def node_weight(node: Dict) -> int:
            """Calculate node weight"""
            if not node:
                return 0
            # Base weight is 1, if there are child nodes, calculate recursively
            weight = 1
            for child in node.get('tokens', []):
                weight += node_weight(child)
            return weight
        
        def compute_edit_distance(node1: Dict, node2: Dict) -> int:
            """Compute edit distance between two AST nodes"""
            # Both nodes are empty
            if not node1 and not node2:
                return 0
            
            # One of the nodes is empty
            if not node1:
                return node_weight(node2)
            if not node2:
                return node_weight(node1)
            
            # Compare key attributes of nodes
            nodes_equal = ASTProcessor._nodes_equal(node1, node2)
            
            tokens1 = node1.get('tokens', [])
            tokens2 = node2.get('tokens', [])
            
            if nodes_equal and not tokens1 and not tokens2:
                # Leaf nodes and equal
                return 0
            
            if nodes_equal:
                # Nodes are equal, compare child nodes
                return ASTProcessor._compute_sequence_edit_distance(tokens1, tokens2)
            else:
                # Nodes are not equal, consider substitute, delete, insert operations
                substitute_cost = 1 + ASTProcessor._compute_sequence_edit_distance(tokens1, tokens2)
                delete_cost = node_weight(node1)
                insert_cost = node_weight(node2)
                
                return min(substitute_cost, delete_cost, insert_cost)
        
        # Calculate edit distance
        distance = compute_edit_distance(ast1, ast2)
        
        # Normalize: divide by the maximum weight of the two ASTs
        weight1 = node_weight(ast1)
        weight2 = node_weight(ast2)
        max_weight = max(weight1, weight2)
        
        if max_weight == 0:
            return 0.0
        
        return min(1.0, distance / max_weight)
    
    @staticmethod
    def _nodes_equal(node1: Dict, node2: Dict) -> bool:
        """Determine if two AST nodes are equal"""
        if not node1 or not node2:
            return False
        
        # Compare node types
        if node1.get('type') != node2.get('type'):
            return False
        
        # Compare token types
        if node1.get('ttype') != node2.get('ttype'):
            return False
        
        # For certain critical node types, compare values
        critical_types = ['Keyword', 'Name', 'Literal']
        if node1.get('type') in critical_types:
            # Normalize values for comparison (ignore case and whitespace)
            val1 = node1.get('value', '').strip().lower()
            val2 = node2.get('value', '').strip().lower()
            return val1 == val2
        
        return True
    
    @staticmethod
    def _compute_sequence_edit_distance(seq1: List[Dict], seq2: List[Dict]) -> int:
        """Compute edit distance between two AST node sequences"""
        m, n = len(seq1), len(seq2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize boundary conditions
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    # Insert j nodes
                    dp[i][j] = sum(ASTProcessor._node_weight_simple(seq2[k]) for k in range(j))
                elif j == 0:
                    # Delete i nodes
                    dp[i][j] = sum(ASTProcessor._node_weight_simple(seq1[k]) for k in range(i))
                else:
                    # Calculate costs of three operations
                    node1, node2 = seq1[i-1], seq2[j-1]
                    
                    # Substitution cost
                    if ASTProcessor._nodes_equal(node1, node2):
                        substitute_cost = ASTProcessor._compute_sequence_edit_distance(
                            node1.get('tokens', []), node2.get('tokens', [])
                        )
                    else:
                        substitute_cost = (ASTProcessor._node_weight_simple(node1) + 
                                         ASTProcessor._node_weight_simple(node2))
                    
                    # Delete and insert costs
                    delete_cost = ASTProcessor._node_weight_simple(node1)
                    insert_cost = ASTProcessor._node_weight_simple(node2)
                    
                    dp[i][j] = min(
                        dp[i-1][j-1] + substitute_cost,  # Substitute
                        dp[i-1][j] + delete_cost,        # Delete
                        dp[i][j-1] + insert_cost         # Insert
                    )
        
        return dp[m][n]
    
    @staticmethod
    def _node_weight_simple(node: Dict) -> int:
        """Simple node weight calculation (non-recursive)"""
        return 1 if node else 0


class ParetoOptimal:
    """Pareto Optimal SQL Generator"""
    
    def __init__(self, db_id: str = None):
        """
        Initialize po
        
        Args:
            db_id: SQLite database name for executability checking
        """
        self.db_id = db_id
        self.ast_processor = ASTProcessor()
        
        # Extended SQL keywords list
        self.sql_keywords = {
            'select', 'from', 'where', 'join', 'inner', 'left', 'right', 'full', 'outer',
            'on', 'and', 'or', 'not', 'in', 'exists', 'like', 'between', 'is', 'null',
            'group', 'by', 'order', 'having', 'limit', 'offset', 'distinct', 'all',
            'union', 'intersect', 'except', 'case', 'when', 'then', 'else', 'end',
            'insert', 'update', 'delete', 'create', 'drop', 'alter', 'table', 'view',
            'index', 'into', 'values', 'set', 'as', 'asc', 'desc', 'count', 'sum',
            'avg', 'min', 'max', 'with', 'recursive', 'over', 'partition', 'window',
            'cast', 'convert', 'substring', 'trim', 'upper', 'lower', 'length',
            'coalesce', 'nullif', 'round', 'floor', 'ceil', 'abs', 'mod', 'power',
            'sqrt', 'log', 'exp', 'sin', 'cos', 'tan', 'concat', 'replace'
        }
    
    def evaluate_executability(self, sql: str) -> float:
        """
        Evaluate SQL executability
        
        Args:
            sql: SQL query string
            
        Returns:
            Executability score (1.0 for executable, 0.0 for non-executable)
        """
        if not self.db_id:
            # If no database, perform simple syntax check
            try:
                parsed = sqlparse.parse(sql)
                return 1.0 if parsed else 0.0
            except:
                return 0.0
        
        try:
            return validate_sql_statement(sql, self.db_id, args.dataset)
        except:
            return 0.0
    
    def _add_limit_to_sql(self, sql: str, limit: int) -> str:
        """Add LIMIT clause to SQL"""
        sql = sql.strip().rstrip(';')
        if 'LIMIT' not in sql.upper():
            sql += f' LIMIT {limit}'
        return sql
    
    def evaluate_schema_conformity(self, sql: str, schema_links: Set[str]) -> float:
        """
        Evaluate schema conformity
        
        Args:
            sql: SQL query string
            schema_links: Set of schema links (table and column names)
            
        Returns:
            Schema conformity score (between 0-1)
        """
        schema_used = self._extract_schema_from_sql(sql)
        
        if not schema_used and not schema_links:
            return 1.0
        
        if not schema_used:
            return 0.0
            
        if not schema_links:
            return 0.0
        
        # Calculate intersection and union
        intersection = schema_used.intersection(schema_links)
        union = schema_used.union(schema_links)
        
        # Use Jaccard similarity
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        
        # Also use coverage: how much of schema_used is in schema_links
        coverage = len(intersection) / len(schema_used) if schema_used else 0.0
        
        # Combine both metrics
        return (jaccard_similarity + coverage) / 2.0
    
    def _extract_schema_from_sql(self, sql: str) -> Set[str]:
        """Extract schema elements (table and column names) used in SQL"""
        schema_elements = set()
        
        # Preprocess SQL: remove string constants to avoid misidentification
        sql_cleaned = self._remove_string_literals(sql)
        
        # Use regex to extract all possible identifiers
        # Match words starting with letters, containing letters, numbers, underscores
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b', sql_cleaned)
        
        # Convert to lowercase and filter keywords
        for word in words:
            word_lower = word.lower()
            if word_lower not in self.sql_keywords:
                schema_elements.add(word_lower)
        
        # Additional processing: extract table names and column names from table.column format
        dot_patterns = re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]*\.[a-zA-Z][a-zA-Z0-9_]*)\b', sql_cleaned)
        for pattern in dot_patterns:
            parts = pattern.split('.')
            for part in parts:
                part_lower = part.lower()
                if part_lower not in self.sql_keywords:
                    schema_elements.add(part_lower)
        
        return schema_elements
    
    def _remove_string_literals(self, sql: str) -> str:
        """Remove string literals from SQL to avoid misidentification"""
        # Remove single-quoted strings
        sql = re.sub(r"'[^']*'", "''", sql)
        # Remove double-quoted strings
        sql = re.sub(r'"[^"]*"', '""', sql)
        # Remove backtick-quoted strings (MySQL style)
        sql = re.sub(r'`[^`]*`', '``', sql)
        return sql

    def _get_db_path(self) -> Optional[str]:
        """Resolve the SQLite file path for the current database id."""
        if not self.db_id:
            return None
        return os.path.join('data', args.dataset, 'database', self.db_id, f'{self.db_id}.sqlite')

    def _execute_sql_with_metadata(self, sql: str) -> Dict[str, Any]:
        """Execute SQL and capture lightweight result metadata."""
        db_path = self._get_db_path()
        if not db_path or not os.path.exists(db_path):
            return {
                'success': False,
                'error': 'database_not_found',
                'rows': [],
                'column_names': [],
            }

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            column_names = [description[0] for description in (cursor.description or [])]
            conn.close()
            return {
                'success': True,
                'error': None,
                'rows': rows,
                'column_names': column_names,
            }
        except Exception as exc:
            try:
                conn.close()
            except Exception:
                pass
            return {
                'success': False,
                'error': str(exc),
                'rows': [],
                'column_names': [],
            }

    def _normalize_result_type(self, row_count: int, column_count: int) -> str:
        """Classify result shape into a coarse result type."""
        if row_count == 0:
            return 'empty'
        if row_count == 1 and column_count == 1:
            return 'scalar'
        if row_count == 1:
            return 'single_row'
        return 'multi_row'

    def _infer_question_intent(self, question: str) -> str:
        """Classify the semantic intent expressed by the question."""
        text = (question or '').lower().strip()
        if not text:
            return 'lookup'
        if any(phrase in text for phrase in ('how many', 'number of', 'count', 'total number')):
            return 'count'
        if any(phrase in text for phrase in ('sum', 'average', 'avg', 'mean', 'maximum', 'minimum', 'max ', 'min ', 'highest', 'lowest', 'top ')):
            return 'aggregate'
        if text.startswith(('is ', 'are ', 'does ', 'do ', 'did ', 'has ', 'have ', 'was ', 'were ', 'can ', 'could ', 'should ', 'whether ')):
            return 'boolean'
        if any(phrase in text for phrase in ('list', 'show', 'display', 'give me', 'return all', 'all ', 'which ', 'what are', 'find all')):
            return 'list'
        return 'lookup'

    def _infer_expected_cardinality(self, question: str, intent: str) -> str:
        """Estimate the expected cardinality class from the question."""
        text = (question or '').lower().strip()
        if intent in ('count', 'aggregate', 'boolean'):
            return 'scalar'
        if any(phrase in text for phrase in ('top 1', 'highest', 'lowest', 'maximum', 'minimum', 'one ', 'single ', 'first ')):
            return 'single_row'
        if any(phrase in text for phrase in ('list', 'show', 'display', 'all ', 'each ', 'per ', 'top ', 'top 3', 'top 5', 'top ten')):
            return 'multi_row'
        return 'single_row'

    def _infer_expected_value_type(self, question: str, intent: str) -> str:
        """Estimate the semantic result type from the question text."""
        text = (question or '').lower()
        if intent in ('count', 'aggregate'):
            return 'numeric'
        if intent == 'boolean' or any(phrase in text for phrase in ('true', 'false', 'yes', 'no', 'exists', 'available', 'active')):
            return 'boolean'
        if any(phrase in text for phrase in ('date', 'year', 'month', 'day', 'birth', 'created', 'updated', 'time')):
            return 'date'
        if any(phrase in text for phrase in ('name', 'title', 'city', 'country', 'state', 'school', 'department', 'category', 'type', 'address', 'email', 'phone', 'zip', 'code', 'id')):
            return 'string'
        if any(phrase in text for phrase in ('salary', 'age', 'price', 'amount', 'rate', 'score', 'count', 'quantity', 'population', 'revenue', 'balance', 'percent', 'ratio', 'duration')):
            return 'numeric'
        return 'string'

    def _infer_actual_value_type(self, rows: List[tuple]) -> str:
        """Infer the dominant value type from execution results."""
        values = [value for row in rows for value in row if value is not None]
        if not values:
            return 'empty'

        numeric_types = (int, float, np.integer, np.floating)
        bool_count = 0
        numeric_count = 0
        date_like_count = 0
        string_count = 0

        for value in values:
            if isinstance(value, bool):
                bool_count += 1
            elif isinstance(value, numeric_types):
                numeric_count += 1
            else:
                value_text = str(value)
                if re.match(r'^\d{4}-\d{2}-\d{2}$', value_text) or re.match(r'^\d{4}/\d{2}/\d{2}$', value_text):
                    date_like_count += 1
                else:
                    string_count += 1

        type_counts = {
            'boolean': bool_count,
            'numeric': numeric_count,
            'date': date_like_count,
            'string': string_count,
        }
        return max(type_counts, key=type_counts.get)

    def _score_intent_result_type(self, expected_type: str, actual_type: str, result_type: str) -> float:
        """Score intent-result type consistency."""
        if actual_type == 'empty':
            return 0.0
        if expected_type == actual_type:
            return 1.0
        if expected_type == 'numeric' and actual_type == 'date':
            return 0.2
        if expected_type == 'string' and actual_type in ('date', 'numeric') and result_type == 'single_row':
            return 0.4
        if expected_type == 'string' and actual_type == 'string':
            return 1.0
        return 0.0

    def _score_cardinality_alignment(self, expected_cardinality: str, row_count: int, column_count: int) -> float:
        """Score result cardinality alignment."""
        result_cardinality = self._normalize_result_type(row_count, column_count)
        if expected_cardinality == 'scalar':
            return 1.0 if result_cardinality == 'scalar' else (0.5 if result_cardinality == 'single_row' else 0.0)
        if expected_cardinality == 'single_row':
            return 1.0 if result_cardinality in ('scalar', 'single_row') else 0.0
        if expected_cardinality == 'multi_row':
            return 1.0 if result_cardinality == 'multi_row' else (0.4 if result_cardinality == 'single_row' else 0.0)
        return 0.0

    def _score_data_type_plausibility(self, expected_type: str, actual_type: str) -> float:
        """Score whether the result data types fit the semantic domain."""
        if actual_type == 'empty':
            return 0.0
        if expected_type == actual_type:
            return 1.0
        compatible_pairs = {
            ('numeric', 'date'),
            ('date', 'string'),
            ('string', 'date'),
        }
        if (expected_type, actual_type) in compatible_pairs:
            return 0.3
        return 0.0

    def _score_result_non_triviality(self, row_count: int, sql: str) -> float:
        """Score whether the result is non-trivial for a query that should match data."""
        if row_count > 0:
            return 1.0

        sql_lower = (sql or '').lower()
        restrictive_signals = sum(
            1 for keyword in ('where', 'join', 'group by', 'having', 'exists', 'in (', 'limit')
            if keyword in sql_lower
        )
        if restrictive_signals >= 2:
            return 0.1
        if restrictive_signals == 1:
            return 0.3
        return 0.5

    def evaluate_semantic_execution_consistency(self, sql: str, question: str) -> float:
        """Evaluate semantic execution consistency from execution metadata and question analysis."""
        execution = self._execute_sql_with_metadata(sql)
        if not execution['success']:
            return 0.0

        rows = execution['rows']
        column_count = len(execution['column_names'])
        row_count = len(rows)
        result_type = self._normalize_result_type(row_count, column_count)
        actual_type = self._infer_actual_value_type(rows)
        intent = self._infer_question_intent(question)
        expected_cardinality = self._infer_expected_cardinality(question, intent)
        expected_type = self._infer_expected_value_type(question, intent)

        intent_score = self._score_intent_result_type(expected_type, actual_type, result_type)
        cardinality_score = self._score_cardinality_alignment(expected_cardinality, row_count, column_count)
        type_score = self._score_data_type_plausibility(expected_type, actual_type)
        non_triviality_score = self._score_result_non_triviality(row_count, sql)

        composite = (
            0.35 * intent_score +
            0.25 * cardinality_score +
            0.20 * type_score +
            0.20 * non_triviality_score
        )

        return float(np.clip(composite, 0.0, 1.0))
    
    def evaluate_example_consistency(self, sql: str, examples: List[str]) -> float:
        """
        Evaluate example consistency
        
        Args:
            sql: Candidate SQL query
            examples: List of example SQL queries
            
        Returns:
            Example consistency score (between 0-1)
        """
        if not examples:
            return 0.0
        
        sql_ast = self.ast_processor.parse_sql_to_ast(sql)
        similarities = []
        
        for example in examples:
            example_ast = self.ast_processor.parse_sql_to_ast(example)
            distance = self.ast_processor.calculate_edit_distance(sql_ast, example_ast)
            similarity = 1.0 - distance  # Convert distance to similarity
            similarities.append(max(0.0, similarity))  # Ensure similarity is non-negative
        
        # Return average similarity
        return sum(similarities) / len(similarities)
    
    def evaluate_sql_candidates(
        self, 
        candidates: List[str], 
        schema_links: Set[str], 
        examples: List[str],
        question: str = ''
    ) -> List[Tuple[SQLCandidate, EvaluationScore]]:
        """
        Evaluate all SQL candidate queries
        
        Args:
            candidates: List of SQL candidate queries
            schema_links: Schema link information
            examples: List of example SQL queries
            
        Returns:
            List of candidate queries and their evaluation scores
        """
        evaluated_candidates = []
        
        for i, sql in enumerate(candidates):
            candidate = SQLCandidate(sql=sql, index=i)
            
            # Evaluate three dimensions
            executability = self.evaluate_executability(sql)
            schema_conformity = self.evaluate_schema_conformity(sql, schema_links)
            example_consistency = self.evaluate_example_consistency(sql, examples)
            semantic_execution_consistency = self.evaluate_semantic_execution_consistency(sql, question)
            
            score = EvaluationScore(
                executability=executability,
                schema_conformity=schema_conformity,
                example_consistency=example_consistency,
                semantic_execution_consistency=semantic_execution_consistency
            )
            
            evaluated_candidates.append((candidate, score))
        
        return evaluated_candidates
    
    def find_pareto_optimal(
        self, 
        evaluated_candidates: List[Tuple[SQLCandidate, EvaluationScore]]
    ) -> List[SQLCandidate]:
        """
        Find Pareto optimal solution set
        
        Args:
            evaluated_candidates: List of evaluated candidate queries
            
        Returns:
            List of Pareto optimal SQL candidate queries
        """
        # First filter out non-executable queries
        executable_candidates = [
            (candidate, score) for candidate, score in evaluated_candidates
            if score.executability > 0.0
        ]
        
        if not executable_candidates:
            return []
        
        pareto_optimal = []
        
        for i, (candidate_i, score_i) in enumerate(executable_candidates):
            is_dominated = False
            
            for j, (candidate_j, score_j) in enumerate(executable_candidates):
                if i == j:
                    continue
                
                # Check if candidate_i is dominated by candidate_j
                if (score_j.schema_conformity >= score_i.schema_conformity and
                    score_j.example_consistency >= score_i.example_consistency and
                    score_j.semantic_execution_consistency >= score_i.semantic_execution_consistency and
                    (score_j.schema_conformity > score_i.schema_conformity or
                     score_j.example_consistency > score_i.example_consistency or
                     score_j.semantic_execution_consistency > score_i.semantic_execution_consistency)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(candidate_i)
        
        return pareto_optimal
    
    def select_final_sql(
        self, 
        candidates: List[str], 
        schema_links: Set[str], 
        examples: List[str],
        question: str = '',
        selection_strategy: str = "balanced"
    ) -> str:
        """
        Select the final SQL query
        
        Args:
            candidates: List of SQL candidate queries
            schema_links: Schema link information
            examples: List of example SQL queries
            selection_strategy: Selection strategy ("balanced", "schema_priority", "example_priority")
            
        Returns:
            The selected final SQL query
        """
        if not candidates:
            return ""
        
        # Evaluate all candidate queries
        evaluated_candidates = self.evaluate_sql_candidates(candidates, schema_links, examples, question)
        
        # Find Pareto optimal solutions
        pareto_optimal = self.find_pareto_optimal(evaluated_candidates)
        
        if not pareto_optimal:
            # If no Pareto optimal solutions, return the first executable query
            for candidate, score in evaluated_candidates:
                if score.executability > 0.0:
                    return candidate.sql
            return candidates[0]  # Final fallback option
        
        if len(pareto_optimal) == 1:
            return pareto_optimal[0].sql
        
        # If there are multiple Pareto optimal solutions, select based on strategy
        best_candidate = None
        best_score = -1.0
        
        for candidate in pareto_optimal:
            # Get corresponding evaluation score
            score = None
            for c, s in evaluated_candidates:
                if c.index == candidate.index:
                    score = s
                    break
            
            if score is None:
                continue
            
            # Calculate combined score based on selection strategy
            if selection_strategy == "balanced":
                combined_score = (
                    score.schema_conformity +
                    score.example_consistency +
                    score.semantic_execution_consistency
                ) / 3.0
            elif selection_strategy == "schema_priority":
                combined_score = (
                    0.5 * score.schema_conformity +
                    0.2 * score.example_consistency +
                    0.3 * score.semantic_execution_consistency
                )
            elif selection_strategy == "example_priority":
                combined_score = (
                    0.2 * score.schema_conformity +
                    0.5 * score.example_consistency +
                    0.3 * score.semantic_execution_consistency
                )
            else:
                combined_score = (
                    score.schema_conformity +
                    score.example_consistency +
                    score.semantic_execution_consistency
                ) / 3.0
            
            if combined_score > best_score:
                best_score = combined_score
                best_candidate = candidate
        
        return best_candidate.sql if best_candidate else pareto_optimal[0].sql


# Usage example and testing
def demo_pareto_optimal_selection():
    """Demonstrate the usage of Pareto optimal selection"""
    
    # Create po instance
    po = ParetoOptimal()
    # po = ParetoOptimal(database_path=db_path)

    
    # Example data
    candidate_sqls = [
        "SELECT name FROM customers WHERE age > 25",
        "SELECT customer_name FROM customer WHERE customer_age > 25",
        "SELECT c.name FROM customers c WHERE c.age > 25 ORDER BY c.name",
        "SELECT * FROM customers WHERE age > 25",
        "SELECT name, age FROM customers WHERE age > 25 AND city = 'New York'"
    ]
    
    schema_links = {"customers", "name", "age", "city", "customer_id"}
    
    example_sqls = [
        "SELECT name FROM employees WHERE salary > 50000",
        "SELECT product_name FROM products WHERE price > 100"
    ]
    
    print("=== AST Parsing Test ===")
    for i, sql in enumerate(candidate_sqls):
        ast_result = po.ast_processor.parse_sql_to_ast(sql)
        print(f"SQL {i+1}: {sql}")
        print(f"AST Type: {ast_result.get('type')}")
        print(f"Number of child nodes: {len(ast_result.get('tokens', []))}")
        print()
    
    print("=== Schema Extraction Test ===")
    for i, sql in enumerate(candidate_sqls):
        extracted = po._extract_schema_from_sql(sql)
        print(f"SQL {i+1}: {sql}")
        print(f"Extracted Schema: {extracted}")
        conformity = po.evaluate_schema_conformity(sql, schema_links)
        print(f"Schema conformity score: {conformity:.3f}")
        print()
    
    # Select final SQL
    final_sql = po.select_final_sql(
        candidates=candidate_sqls,
        schema_links=schema_links,
        examples=example_sqls,
        selection_strategy="balanced"
    )
    
    print("=== Complete Evaluation Results ===")
    print("Candidate SQL queries:")
    for i, sql in enumerate(candidate_sqls):
        print(f"{i+1}. {sql}")
    
    print(f"\nSelected final SQL: {final_sql}")
    
    # Show detailed evaluation information
    evaluated = po.evaluate_sql_candidates(candidate_sqls, schema_links, example_sqls)
    pareto_optimal = po.find_pareto_optimal(evaluated)
    
    print("\nDetailed evaluation results:")
    for candidate, score in evaluated:
        print(f"SQL {candidate.index + 1}:")
        print(f"  Executability: {score.executability:.3f}")
        print(f"  Schema conformity: {score.schema_conformity:.3f}")
        print(f"  Example consistency: {score.example_consistency:.3f}")
        print()
    
    print("Pareto optimal solutions:")
    for candidate in pareto_optimal:
        print(f"  SQL {candidate.index + 1}: {candidate.sql}")


if __name__ == "__main__":
    demo_pareto_optimal_selection()
