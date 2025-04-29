import matplotlib.pyplot as plt
import numpy as np

# --- Data from your output ---
loss_data_text = """
1	1.447000
2	1.341400
3	1.364900
4	1.422800
5	1.412200
6	1.332200
7	1.308500
8	1.255900
9	1.212400
10	1.239700
11	1.251700
12	1.174000
13	1.155900
14	1.164300
15	1.139900
16	1.160300
17	1.216000
18	1.196700
19	1.115700
20	1.139700
21	1.159100
22	1.156400
23	1.088400
24	1.185400
25	1.096000
26	1.086500
27	1.156500
28	1.143100
29	1.081300
30	1.092300
31	1.074900
32	1.116500
33	1.169100
34	1.043200
35	1.097700
36	1.047700
37	1.073800
38	1.122400
39	1.004300
40	1.029100
41	1.068300
42	1.005800
43	1.013900
44	1.058900
45	1.091600
46	1.083000
47	1.067200
48	0.990100
49	1.065500
50	0.967800
51	0.978200
52	1.102800
53	0.992600
54	0.984200
55	1.007600
56	0.944800
57	1.007900
58	0.933300
59	1.074400
60	0.933500
61	0.970100
62	0.966100
63	0.946800
64	1.017500
65	1.010300
66	0.960100
67	0.937900
68	0.998100
69	0.937600
70	1.000300
71	0.997000
72	0.919500
73	0.914100
74	0.927400
75	1.018600
76	1.060800
77	0.986800
78	0.922500
79	0.978000
80	0.948000
81	1.052500
82	0.983800
83	0.909000
84	1.049800
85	0.926100
86	0.933100
87	0.830400
88	0.963100
89	0.921100
90	0.891600
91	0.915300
92	0.926400
93	0.980100
94	0.962300
95	0.969100
96	0.897800
97	0.958400
98	0.907500
99	0.937900
100	0.793000
101	0.982100
102	0.906700
103	0.938500
104	0.887400
105	0.928000
106	0.842600
107	0.935400
108	0.870100
109	0.826200
110	0.845900
111	0.933200
112	0.863600
113	0.816100
114	0.852000
115	0.823500
116	0.807600
117	0.774100
118	0.890500
119	0.784200
120	0.828100
121	0.793800
122	0.872000
123	0.910000
124	0.908500
125	0.813800
126	0.735700
127	0.717400
128	0.780700
129	0.758400
130	0.763600
131	0.774800
132	0.781200
133	0.803000
134	0.772400
135	0.833100
136	0.890600
137	0.741000
138	0.845000
139	0.797900
140	0.773500
141	0.832600
142	0.877000
143	0.803400
144	0.812800
145	0.773100
146	0.825100
147	0.787200
148	0.734100
149	0.848700
150	0.820700
151	0.785700
152	0.706100
153	0.736000
154	0.833300
155	0.710000
156	0.780300
157	0.766200
158	0.702800
159	0.777700
160	0.720500
161	0.873300
162	0.808400
163	0.813200
164	0.674700
165	0.811100
166	0.883400
167	0.689100
168	0.888700
169	0.825800
170	0.788700
171	0.809800
172	0.696400
173	0.834600
174	0.745100
175	0.763700
176	0.720300
177	0.775200
178	0.760000
179	0.860400
180	0.802700
181	0.786600
182	0.800000
183	0.794800
184	0.781700
185	0.731000
186	0.767500
187	0.744000
188	0.779800
189	0.792000
190	0.732200
191	0.779300
192	0.759300
193	0.724800
194	0.694800
195	0.773200
196	0.801300
197	0.807100
198	0.684900
199	0.700000
200	0.731400
201	0.735100
202	0.779600
203	0.715800
204	0.782600
205	0.800500
206	0.725200
207	0.772300
208	0.768000
209	0.802700
210	0.679300
211	0.734300
212	0.784700
213	0.729800
214	0.796200
215	0.550900
216	0.665700
217	0.585300
218	0.625800
219	0.609900
220	0.599500
221	0.551000
222	0.643200
223	0.657200
224	0.663000
225	0.665200
226	0.599500
227	0.649500
228	0.412100
229	0.727300
230	0.759300
231	0.624700
232	0.655600
233	0.667700
234	0.682700
235	0.656100
236	0.583500
237	0.643800
238	0.643600
239	0.620400
240	0.600900
241	0.638200
242	0.540200
243	0.613100
244	0.555900
245	0.565100
246	0.537900
247	0.639600
248	0.684400
249	0.616000
250	0.588900
251	0.633700
252	0.645700
253	0.612800
254	0.676100
255	0.694500
256	0.664600
257	0.673100
258	0.617900
259	0.662100
260	0.526000
261	0.660400
262	0.567200
263	0.557800
264	0.591500
265	0.663200
266	0.623000
267	0.627600
268	0.589900
269	0.636400
270	0.696400
271	0.643500
272	0.544000
273	0.636400
274	0.632500
275	0.685300
276	0.579700
277	0.562000
278	0.629500
279	0.625700
280	0.540400
281	0.631800
282	0.591400
283	0.666400
284	0.583600
285	0.616200
286	0.516700
287	0.590200
288	0.554300
289	0.618800
290	0.571500
291	0.646100
292	0.547200
293	0.598700
294	0.584000
295	0.677600
296	0.545100
297	0.684900
298	0.619300
299	0.543100
300	0.606400
301	0.571300
302	0.541600
303	0.495300
304	0.636400
305	0.637200
306	0.621500
307	0.660000
308	0.590600
309	0.547600
310	0.562000
311	0.595200
312	0.622000
313	0.540300
314	0.642200
315	0.544500
316	0.589900
317	0.603000
318	0.507300
319	0.620600
320	0.503200
321	0.635300
322	0.484100
323	0.557400
324	0.549100
325	0.456800
326	0.547700
327	0.480300
328	0.486000
329	0.570100
330	0.519500
331	0.447100
332	0.469300
333	0.514400
334	0.494700
335	0.472300
336	0.580700
337	0.529100
338	0.512900
339	0.385900
340	0.533900
341	0.550800
342	0.471600
343	0.448800
344	0.457200
345	0.466200
346	0.554400
347	0.499100
348	0.573000
349	0.547700
350	0.477500
351	0.522500
352	0.547300
353	0.540200
354	0.464200
355	0.608500
356	0.480300
357	0.484300
358	0.538000
359	0.487300
360	0.516400
361	0.501700
362	0.479600
363	0.425000
364	0.478500
365	0.429500
366	0.475800
367	0.483300
368	0.496100
369	0.429800
370	0.444200
371	0.437400
372	0.519900
373	0.494400
374	0.499000
375	0.500300
376	0.427700
377	0.480200
378	0.499600
379	0.371700
380	0.472700
381	0.458100
382	0.405900
383	0.482400
384	0.396700
385	0.474500
386	0.489500
387	0.433400
388	0.476700
389	0.486600
390	0.519500
391	0.453200
392	0.481600
393	0.515300
394	0.468300
395	0.473300
396	0.551300
397	0.469700
398	0.442100
399	0.451900
400	0.389000
401	0.449600
402	0.420600
403	0.533800
404	0.459300
405	0.498000
406	0.517800
407	0.493500
408	0.554900
409	0.450300
410	0.556400
411	0.428000
412	0.483800
413	0.579700
414	0.427800
415	0.489000
416	0.443500
417	0.405500
418	0.522200
419	0.456600
420	0.442700
421	0.447100
422	0.484400
423	0.433500
424	0.540500
425	0.410700
426	0.442300
427	0.432400
428	0.541600
"""

# Parse the data into lists of steps and losses
lines = loss_data_text.strip().split('\n')
steps = []
losses = []
for line in lines:
    parts = line.split()
    if len(parts) == 2:
        try:
            steps.append(int(parts[0]))
            losses.append(float(parts[1]))
        except ValueError:
            print(f"Skipping invalid line: {line}") # Handle potential header/footer lines if any

# --- Plotting ---
num_epochs = 4
total_steps = steps[-1] if steps else 0
if total_steps == 0 or num_epochs == 0:
    print("Error: No steps or epochs found.")
else:
    steps_per_epoch = total_steps // num_epochs # Use integer division for calculation

    plt.figure(figsize=(15, 6)) # Make the plot wider
    plt.plot(steps, losses, label='Training Loss', alpha=0.8)

    # Add vertical lines and text to mark epoch boundaries
    for i in range(1, num_epochs):
        epoch_end_step = i * steps_per_epoch
        # Find the closest actual step to the calculated boundary
        closest_step_index = min(range(len(steps)), key=lambda idx: abs(steps[idx] - epoch_end_step))
        actual_epoch_end_step = steps[closest_step_index]

        plt.axvline(x=actual_epoch_end_step, color='red', linestyle='--', linewidth=1, label=f'End of Epoch {i}' if i == 1 else "")
        plt.text(actual_epoch_end_step + 5, max(losses) * 0.95, f'Epoch {i+1} begins', rotation=90, verticalalignment='top', color='red')

    # Add a text label for the start of Epoch 1
    plt.text(steps[0] + 5, max(losses) * 0.95, 'Epoch 1 begins', rotation=90, verticalalignment='top', color='blue')


    plt.xlabel("Training Step")
    plt.ylabel("Training Loss")
    plt.title("Training Loss per Step with Epoch Boundaries")
    plt.legend(loc='upper right') # Adjust legend location if needed
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=0) # Start y-axis at 0
    plt.xlim(left=0)   # Start x-axis at 0
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()

    # --- Alternative: Average Loss Per Epoch ---
    avg_losses_per_epoch = []
    epoch_numbers = list(range(1, num_epochs + 1))

    for i in range(num_epochs):
        start_index = i * steps_per_epoch
        # Find the actual start step index
        start_step_index = min(range(len(steps)), key=lambda idx: abs(steps[idx] - start_index -1)) # -1 because steps start at 1
        if steps[start_step_index] < start_index + 1: # ensure we start at or after the theoretical start
             start_step_index += 1
        start_step_index = max(0, start_step_index) # clamp at 0

        # Find the actual end step index
        end_index = (i + 1) * steps_per_epoch
        end_step_index = min(range(len(steps)), key=lambda idx: abs(steps[idx] - end_index))
        if steps[end_step_index] > end_index and end_step_index > 0: # ensure we end at or before the theoretical end
            end_step_index -=1
        end_step_index = min(len(losses) - 1, end_step_index) # clamp at max index

        # Extract losses for the current epoch based on actual step indices
        # +1 because slicing is exclusive at the end
        epoch_losses = losses[start_step_index : end_step_index + 1]

        if epoch_losses: # Avoid division by zero if a chunk is empty
             avg_losses_per_epoch.append(np.mean(epoch_losses))
        else:
             avg_losses_per_epoch.append(np.nan) # Or handle as appropriate


    print("\nAverage Loss Per Epoch:")
    for epoch, avg_loss in zip(epoch_numbers, avg_losses_per_epoch):
        print(f"Epoch {epoch}: {avg_loss:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_numbers, avg_losses_per_epoch, marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.title("Average Training Loss per Epoch")
    plt.xticks(epoch_numbers) # Ensure ticks are at 1, 2, 3, 4
    plt.grid(True)
    plt.ylim(bottom=0) # Start y-axis at 0
    plt.tight_layout()
    plt.show()