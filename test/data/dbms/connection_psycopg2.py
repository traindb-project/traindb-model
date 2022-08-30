import psycopg2
import pandas as pd
import copy

def print_conditions(conditions, seperator='Λ'):
    """Pretty prints a set of conditions with a custom seperator."""

    formula = ""
    for i, (table, condition) in enumerate(conditions):
        formula += table + "." + condition
        if i < len(conditions) - 1:
            formula += ' ' + seperator + ' '

    return formula


def gen_full_join_query(schema_graph, relationship_set, table_set, join_type):
    """
    Creates the full outer join to for a relationship set for join_type FULL OUTER JOIN or JOIN
    """

    from_clause = ""
    if len(relationship_set) == 0:
        assert(len(table_set) == 1)

        from_clause = list(table_set)[0]

    else:
        included_tables = set()
        relationships = copy.copy(relationship_set)

        while relationships:
            # first relation to be included
            if len(included_tables) == 0:
                relationship = relationships.pop()
                relationship_obj = schema_graph.relationship_dictionary[relationship]
                included_tables.add(relationship_obj.start)
                included_tables.add(relationship_obj.end)
                from_clause += relationship_obj.start + " " + join_type + " " + relationship_obj.end + " ON " + relationship
            else:
                # search in suitable relations
                relationship_to_add = None
                for relationship in relationships:
                    relationship_obj = schema_graph.relationship_dictionary[relationship]
                    if (relationship_obj.start in included_tables and relationship_obj.end not in included_tables) or \
                            (relationship_obj.end in included_tables and relationship_obj.start not in included_tables):
                        relationship_to_add = relationship
                if relationship_to_add is None:
                    raise ValueError("Query not a tree")
                # add it to where formula
                relationship_obj = schema_graph.relationship_dictionary[relationship_to_add]
                if (relationship_obj.start in included_tables and relationship_obj.end not in included_tables):
                    from_clause += " " + join_type + " " + relationship_obj.end + " ON " + relationship_to_add
                    included_tables.add(relationship_obj.end)
                    relationships.remove(relationship_to_add)
                elif (relationship_obj.end in included_tables and relationship_obj.start not in included_tables):
                    from_clause += " " + join_type + " " + relationship_obj.start + " ON " + relationship_to_add
                    included_tables.add(relationship_obj.start)
                    relationships.remove(relationship_to_add)

    return "SELECT {} FROM " + from_clause + " {}"

class DBConnection:

    def __init__(self, db_user="postgres", db_password="postgres", db_host="localhost", db_port="5432", db="shopdb"):
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.db = db

    def vacuum(self):
        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        old_isolation_level = connection.isolation_level
        connection.set_isolation_level(0)
        query = "VACUUM"
        cursor = connection.cursor()
        cursor.execute(query)
        connection.commit()
        connection.set_isolation_level(old_isolation_level)

    def get_dataframe(self, sql):
        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        return pd.read_sql(sql, connection)

    def submit_query(self, sql):
        """Submits query and ignores result."""

        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        cursor = connection.cursor()
        cursor.execute(sql)
        connection.commit()

    def get_result(self, sql):
        """Fetches exactly one row of result set."""

        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        cursor = connection.cursor()

        cursor.execute(sql)
        record = cursor.fetchone()
        result = record[0]

        if connection:
            cursor.close()
            connection.close()

        return result

    def get_result_set(self, sql, return_columns=False):
        """Fetches all rows of result set."""

        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        cursor = connection.cursor()

        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        if connection:
            cursor.close()
            connection.close()

        if return_columns:
            return rows, columns

        return rows

    def true_cardinality(self, query):
        full_join_query = gen_full_join_query(self.schema_graph, query.relationship_set, query.table_set, "JOIN")

        where_cond = print_conditions(query.conditions, seperator='AND')
        if where_cond != "":
            where_cond = "WHERE " + where_cond
        sql_query = full_join_query.format("COUNT(*)", where_cond)
        cardinality = self.db_connection.get_result(sql_query)
        return sql_query, cardinality
