# === Apply Media Tier Logic ===
def apply_media_tier_logic(df):
    try:
        file_id = "1LIcEKO-fdXfo1v-IUeU64He5mh6ti1nc"  # ID file Excel update Media Tier

        # Download file Excel
        download_url_media_tier = f"https://drive.google.com/uc?id={file_id}"
        response_media_tier = requests.get(download_url_media_tier)
        xls_media_tier = pd.ExcelFile(BytesIO(response_media_tier.content))

        # Load the relevant sheets
        sheet_client = pd.read_excel(xls_media_tier, sheet_name="Le Minerale - from Client")
        sheet_online = pd.read_excel(xls_media_tier, sheet_name="Online with AVE - Updated")
        
        load_success = True
    except Exception as e:
        st.error(f"❌ Gagal load file Media Tier dari Google Drive: {e}")
        load_success = False

    if load_success:

        # Mapping 'Media Name' to 'Media Tier' for both sheets
        media_tier_dict = {}
        
        # Get media tier from the "Le Minerale - from Client"
        for index, row in sheet_client.iterrows():
            media_name = row['Media Name']
            media_tier = row['Media Tier']
            media_tier_dict[media_name] = media_tier

        # Get media tier from the "Online with AVE - Updated"
        for index, row in sheet_online.iterrows():
            media_name = row['Media Name']
            media_tier = row['Media Tier']
            if media_name not in media_tier_dict:
                media_tier_dict[media_name] = media_tier

        # Apply media tier to the raw data
        for index, row in df[df['Media Name'].notna()].iterrows():
            media_name = row['Media Name']
            
            # Check if Media Name is in our media_tier_dict, otherwise apply Ad Value logic
            if media_name in media_tier_dict:
                df.at[index, 'Media Tier'] = media_tier_dict[media_name]
            else:
                ad_value = row['Ad Value']
                if ad_value >= 18000000:
                    df.at[index, 'Media Tier'] = 1
                elif ad_value >= 12600000:
                    df.at[index, 'Media Tier'] = 2
                else:
                    df.at[index, 'Media Tier'] = 3
        return df
    
    else:
        st.stop()


            #st.info(f"🕒 Proses ini berjalan selama {int(hours)} jam {int(minutes)} menit {int(seconds)} detik.")


            # 1. Tampilkan Summary Execution Report
            #st.subheader("📊 Summary Execution Report")
            #with st.expander("Lihat Summary Execution Report"):
            #    if not summary_df.empty:
            #        # Hilangkan .0 di Output Value
            #        summary_cleaned = summary_combined.copy()
            #        summary_cleaned["Output Value"] = summary_cleaned["Output Value"].astype(str).str.replace(r"\.0$", "", regex=True)

            #       summary_sorted = summary_cleaned.sort_values(by="Priority", ascending=False)
            #       st.dataframe(summary_sorted[[
            #           "Priority", "Matching Column", "Matching Value", "Matching Type",
            #           "Channel", "Affected Rows", "Output Column", "Output Value"
            #       ]])

            #      # 🔹 Tambahan Ringkasan Noise Tag
            #        if "Noise Tag" in df_processed.columns:
            #            st.markdown("**🧮 Ringkasan Noise Tag:**")
            #            noise_summary = df_processed["Noise Tag"].astype(str).str.replace(r"\.0$", "", regex=True).value_counts().reset_index()
            #            noise_summary.columns = ["Noise Tag", "Jumlah"]

            #            description_map = {
            #                "0": "Valid Content",
            #                "1": "Official Brand",
            #                "2": "Exclude Content (noise)",
            #                "3": "Need QC"
            #            }
            #            noise_summary["Description"] = noise_summary["Noise Tag"].astype(str).map(description_map)
            #            st.dataframe(noise_summary)
            #    else:
            #        st.info("ℹ️ Tidak ada rule yang match pada data ini.")

            # 2. Tampilkan Chain Overwrite Tracker
            #st.subheader("🧩 Chain Overwrite Tracker")
            #with st.expander("Lihat Chain Overwrite Tracker"):
            #    chain_overwrite_columns = [output_column + " - Chain Overwrite" for output_column in ["Noise Tag"]]
            #    if any(col in df_final.columns for col in chain_overwrite_columns):
            #        chain_overwrite_df = df_final[chain_overwrite_columns]
            #        st.dataframe(chain_overwrite_df)
            #    else:
            #        st.info("ℹ️ Tidak ada perubahan tercatat (Chain Overwrite kosong).")

            # 3. Tombol Download Hasil di paling bawah

            # Simpan nama file hasil ke session_state
            #st.session_state["output_filename"] = output_filename

            #st.success(f"⏱️ Proses selesai dalam {int(minutes)} menit {int(seconds)} detik")
            #st.download_button(
            #    label="⬇️ Download Hasil Excel",
            #    data=open(output_filename, "rb").read(),
            #    file_name=output_filename,
            #    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            #)

    # Tampilkan tombol download jika tersedia
    #if "output_excel_bytes" in st.session_state:
    #    st.download_button(
    #        label="⬇️ Download Hasil Excel",
    #        data=st.session_state["output_excel_bytes"],
    #        file_name=st.session_state["download_filename"],
    #        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    #    )