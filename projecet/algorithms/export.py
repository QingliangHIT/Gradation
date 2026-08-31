import pandas as pd



def export_excel(
        df,
        filename
):


    with pd.ExcelWriter(
        filename
    ) as writer:


        df.to_excel(
            writer,
            sheet_name="grading",
            index=False
        )

