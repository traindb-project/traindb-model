# JDBC connection test

import jaydebeapi

conn = jaydebeapi.connect("org.hsqldb.jdbcDriver",
                           "jdbc:hsqldb:mem:.",
                           {'user': "SA", 'password': "",
                            'other_property': "foobar"},
                           "/path/to/hsqldb.jar",)

curs = conn.cursor()
curs.execute("select * from CUSTOMER")
curs.fetchall()

curs.close()
conn.close()
