import pandas as pd
meta = pd.read_excel("data/hack.xlsx", sheetname="POX Process Flowsheet (Block)")
sheets = ["POX Process Flowsheet (Block)", "POX Process Flowsheet",
         "Schematic - Feed", "Schematic - HX", "Schematic - Autoclave DCS (e.g)",
         "Schematic - Autoclave DCS (e.g)"]

with open("strings.csv", 'w') as f:
    f.write("column, desc, unit")
    for sheet in sheets:
        meta = pd.read_excel("data/hack.xlsx", sheetname=sheet)
        for i in range(len(meta)):
            for j, c in enumerate(meta.columns):
                try:
                    datum = meta.iloc[i, j]
                    datum = str(datum)[9:]
                    if '\\' in str(datum):
                        try:
                            unit = str(meta.iloc[i, j+2])
                        except UnicodeEncodeError:
                            unit = ""
                        if unit == "nan":
                            unit = ""
                        f.write("%s, %s, %s\n" % (datum, meta.iloc[i, j+1], unit))
                except:
                    pass

