with open('AudioDenoiser/denoiser_pipeline.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_calc = '''        st = self.stats
        st.in_dbfs = float(10 * np.log10(ein))
        st.out_dbfs = float(10 * np.log10(eout))'''

new_calc = '''        st = self.stats
        # in_dbfs is now calculated at the very beginning of process_block
        st.out_dbfs = float(10 * np.log10(eout))'''


content = content.replace(old_calc, new_calc)

old_blocks = '''        st.infer_ms = infer_ms
        st.blocks += 1'''
new_blocks = '''        st.infer_ms = infer_ms
        # st.blocks is now incremented in process_block'''
content = content.replace(old_blocks, new_blocks)

with open('AudioDenoiser/denoiser_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Replaced!")
